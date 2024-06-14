import os
import time
import ast
import warnings
import torch
import numpy as np
import torchio as tio

from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold

from data.argumentation import get_transform_aug
from data.load import load_data, preprocess
from utils.arguments import parse_arguments
from utils.utils import seed_everything, get_hashes, print_results, output_results, save_cvs

P_MODE_LIST = {'cnn': -1, 'protopnet': 0, 'xprotonet': 1, 'mprotonet': 2, 'maprotonet': 3}


def main():
    tic = time.time()
    # 1. parse arguments
    args = parse_arguments()
    args.device_id = args.gpus = args.local_rank
    args.device = torch.device('cuda:' + str(args.device_id))
    # str
    args.model_name = args.model_name.lower()
    args.data_name = \
        args.data_path[args.data_path.find('BraTS'):args.data_path.find('BraTS') + 10].replace('_', '').replace('/', '')
    # bool
    args.augmented = args.augmented == 1
    args.mmloss = args.mmloss == 1
    args.save_model = args.save_model == 1
    args.use_amp = args.use_amp == 1
    args.use_da = args.use_da == 1
    # structural
    args.prototype_shape = ast.literal_eval(args.prototype_shape)
    args.coefs = ast.literal_eval(args.coefs)
    # others
    assert args.batch_size % int(os.environ['n_gpus']) == 0, 'Your batch size can not be divided by the number of GPUs'
    args.batch_size = args.batch_size // int(os.environ['n_gpus'])
    if isinstance(args.coefs, list):
        args.coefs = args.coefs[0]
    print(args)

    # 2. initial setting
    # multithreading
    torch.set_num_threads(args.n_threads)
    # warning
    warnings.filterwarnings('ignore', message="The epoch parameter in `scheduler.step\(\)` was not")
    # hash code
    opts_hash = get_hashes(args)  # for saving dir
    # seed
    seed_everything(args.seed)
    # deterministic
    if not args.use_da:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 3. data
    # preload
    x, y = load_data(data_path=args.data_path)
    # normal transform for preprocess
    transform = [tio.ToCanonical(), tio.CropOrPad(target_shape=(192, 192, 144))]
    transform += [tio.Resample(target=(1.5, 1.5, 1.5))]
    transform += [tio.ZNormalization()]
    transform = tio.Compose(transform)
    # pre-process
    dataset = tio.SubjectsDataset(list(x), transform=transform)
    data_loader = DataLoader(dataset, num_workers=args.n_workers)
    x = preprocess(data_loader)
    del dataset, data_loader, transform
    # data augmentation (only training)
    if args.augmented:
        if args.aug_seq is None:
            args.aug_seq = 'af0-no0-bl0-br0-co0-an0-gi0-ga0-fl0'
        transform_aug = get_transform_aug(aug_seq=args.aug_seq)
    else:
        transform_aug = []
    transform_train = tio.Compose(transform_aug) if args.augmented else None
    transform_test = None
    del transform_aug
    toc = time.time()
    print(f"Elapsed time is {toc - tic:.6f} seconds.")

    # 4. 5-fold CV settings
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=args.seed)
    cv_fold = cv.get_n_splits()
    splits = np.zeros(y.shape[0], dtype=int)

    # 5. CV training
    args.p_mode = P_MODE_LIST[args.model_name]
    torch.save({'x': x, 'y': y, 'transform_train': transform_train, 'transform_test': transform_test, 'args': args}, 'data.pt')
    torch.save({'f_x':  np.zeros(y.shape), 'lcs': {}, 'n_prototypes': np.zeros((cv_fold, np.max(y.argmax(axis=-1)) + 1)), 'iads': {}}, 
            'metrics.pt')
    for i, (I_train, I_test) in enumerate(cv.split(x, y.argmax(1))):
        print(f">>>>>>>> CV = {i + 1}:")
        splits[I_test] = i
        torch.save({'I_train': I_train, 'I_test': I_test}, 'index.pt')
        script = f'python -m torch.distributed.launch --nproc_per_node {os.environ["n_gpus"]} --nnodes 1 --node_rank 0 ' \
                 f'./train_one_fold.py --cv-i {i} --opts-hash {opts_hash}'
        os.system(script)
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
        print()
    metrics = torch.load('metrics.pt')
    f_x, lcs, n_prototypes, iads = metrics['f_x'], metrics['lcs'], metrics['n_prototypes'], metrics['iads']
    for f in ['index.pt', 'data.pt', 'metrics.pt']:
        os.remove(f)

    # 6. overall evaluation
    if args.local_rank == 0:
        print(f">>>>>>>> {cv_fold}-fold CV Results:")
        print_results("Test", f_x, y, lcs, n_prototypes, iads, splits)
        output_results(args.data_name, args, f_x, y, lcs, n_prototypes, iads, splits)
        cv_dir = './results/cvs/'
        if not os.path.exists(cv_dir):
            os.makedirs(cv_dir)
        save_cvs(cv_dir, args, f_x, y, lcs, iads, splits)
        print("Finished.")
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
        print()


if __name__ == "__main__":
    main()

