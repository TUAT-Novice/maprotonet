import os
import time
import ast
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchio as tio
import numpy as np
mp.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from thop import profile

from models.protos import CNN, ProtoNets
from train import FocalLoss, train, test
from data.argumentation import get_transform_aug
from data.load import load_data, preprocess
from utils.arguments import parse_arguments
from utils.metrics import process_iad
from utils.push import push_prototypes
from utils.utils import seed_everything, print_main, get_hashes, print_param, print_results, output_results, save_cvs

P_MODE_LIST = {'cnn': -1, 'protopnet': 0, 'xprotonet': 1, 'mprotonet': 2, 'maprotonet': 3}


def train_one_fold(
        local_rank, args, x, y, I_train, I_test, cv_i, transform_train, transform_test,
        f_x,  lcs, n_prototypes, iads, opts_hash, cv_fold=5
):
    # 1. ddp initialize
    os.environ['LOCAL_RANK'] = os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo', init_method='env://')
    torch.cuda.set_device(local_rank)
    # 2. dataset
    dataset_train = tio.SubjectsDataset(list(x[I_train]), transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.n_workers,
                              sampler=sampler_train, pin_memory=True, drop_last=True)
    
    if local_rank == 0:
        dataset_test = tio.SubjectsDataset(list(x[I_test]), transform=transform_test)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        loader_test = DataLoader(dataset_test, batch_size=args.batch_size // 2, num_workers=args.n_workers,
                sampler=sampler_test, pin_memory=True)
        if args.p_mode >= 0:
            dataset_push = tio.SubjectsDataset(list(x[I_train]), transform=transform_test)
            sampler_push = torch.utils.data.SequentialSampler(dataset_push)
            loader_push = DataLoader(dataset_push, batch_size=args.batch_size // 2, num_workers=args.n_workers,
                    sampler=sampler_push, pin_memory=True)
    # 3. model
    in_size = (4,) + dataset_train[0]['t1']['data'].shape[1:]
    out_size = dataset_train[0]['label'].shape[0]
    kwargs = {'in_size': in_size, 'out_size': out_size, 'backbone': args.backbone, 'n_res_block': args.n_res_block,
              'prototype_shape': args.prototype_shape, 'f_dist': args.f_dist, 'p_mode': args.p_mode,
              }
    if args.p_mode == 0:  # for protopnet
        kwargs['topk_p'] = args.topk_p
    if args.p_mode == 3:  # for maprotonet
        kwargs['n_scales'] = args.n_scales
        kwargs['scale_mode'] = args.scale_mode
    net = ProtoNets(**kwargs).to(local_rank) if args.p_mode >= 0 else CNN(**kwargs).to(local_rank)
    # load
    if args.load_model is not None:
        if args.load_model.startswith(args.model_name):
            model_name_i = f'{args.load_model}_cv{cv_i}'
            model_path_i = f'./results/models/{model_name_i}.pt'
        else:
            model_name_i = f'{args.load_model[args.load_model.find(args.model_name):]}_cv{i}'
            model_path_i = f'{args.load_model}_cv{i}.pt'
        net.load_state_dict(torch.load(model_path_i))
        print_main(f"Load Model {args.model_name} from {model_path_i}", local_rank)
    else:
        model_name_i = f'{args.model_name}_{opts_hash}_cv{cv_i}'
    net = net.to(local_rank)
    # num of params, flops
    input_x = torch.randn(in_size).unsqueeze(0).to(local_rank)
    net.flops, net.params = profile(net, inputs=(input_x,))
    if local_rank == 0:
        print_param(net, show_each=False)
        print(f"Model: {model_name_i}\n{str(net)}")
        print(f"Hyper-parameters = {args}")
        print(f"Number of Epoch = {args.epoch}")
    del input_x
    # model ddp
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    dist.barrier()

    # 4. saving path
    img_dir = f'./results/saved_imgs/{model_name_i}/'
    if local_rank == 0 and not os.path.exists(img_dir):
        os.makedirs(img_dir)
    prototype_img_filename_prefix = 'prototype-img'
    proto_bound_boxes_filename_prefix = 'bb'
    dist.barrier()

    # 5. optimizer
    if args.p_mode >= 0:
        params = [
            {'params': net.module.backbone.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            {'params': net.module.features_add_ons.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            {'params': net.module.prototype_vectors, 'lr': args.lr, 'weight_decay': 0},
        ]
        if args.p_mode >= 1:
            params += [
                {'params': net.module.p_map.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            ]
        if args.p_mode >= 3:
            params += [
                {'params': net.module.scale_layers.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            ]
        params_last_layer = [
            {'params': net.module.last_layer.parameters(), 'lr': args.lr, 'weight_decay': 0},
        ]
        if args.op_opt == 'Adam':
            optimizer = optim.Adam(params)
            optimizer_last_layer = optim.Adam(params_last_layer)
        elif args.op_opt == 'AdamW':
            optimizer = optim.AdamW(params)
            optimizer_last_layer = optim.AdamW(params_last_layer)
        else:
            optimizer = optim.SGD(params, momentum=0.9)
            optimizer_last_layer = optim.SGD(params_last_layer, momentum=0.9)
    else:
        if args.op_opt == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.op_opt == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # 6. warmup & lr scheduler
    if 'WU' in args.lr_opt and args.wu_n <= 0:
        args.wu_n = args.epoch // 5
    if 'StepLR' in args.lr_opt:
        if args.lr_n <= 0:
            args.lr_n = args.epoch // 10
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_n)
    elif 'CosALR' in args.lr_opt:
        if args.lr_n <= 0:
            args.lr_n = args.epoch - args.wu_n if 'WU' in args.lr_opt else args.epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_n)
    if 'WU' in args.lr_opt:
        scheduler0 = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / args.wu_n)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler0, scheduler], [args.wu_n])

    # 7. balanced classification
    if args.bc_opt in ['BCE', 'BFL']:
        class_weight = torch.FloatTensor(1 / y[I_train].sum(0))
    elif args.bc_opt in ['B2CE', 'B2FL']:
        class_weight = torch.FloatTensor(1 / y[I_train].sum(0) ** 0.5)
    elif args.bc_opt in ['CBCE', 'CBFL']:
        beta = 1 - 1 / y[I_train].sum()
        class_weight = torch.FloatTensor((1 - beta) / (1 - beta ** y[I_train].sum(0)))
    else:
        class_weight = torch.ones(y[I_train].shape[1])
    if 'FL' in args.bc_opt:
        criterion = FocalLoss(weight=class_weight).to(local_rank)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(local_rank)

    # 8. training
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    if args.load_model is None:
        print_main("Epoch =", local_rank, end='', flush=True)
        for e in range(args.epoch):
            print_main(f" {e + 1}", local_rank, end='', flush=True)
            if args.lr_opt != 'Off':
                print_main(f"(lr={scheduler.get_last_lr()[0]:g})", local_rank, end='', flush=True)
            # stage 1
            train(net, loader_train, optimizer, criterion, scaler, args, local_rank, stage='joint', class_weight=class_weight)
            if args.lr_opt != 'Off':
                scheduler.step()
            # stage 2
            if args.p_mode >= 0 \
                    and e + 1 >= 10 \
                    and e + 1 in [i for i in range(args.epoch + 1) if i % 10 == 0]:
                if local_rank == 0:
                    push_prototypes(
                        loader_push,
                        net.module,
                        args,
                        root_dir_for_saving_prototypes=None,
                        prototype_img_filename_prefix=prototype_img_filename_prefix,
                        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix
                    )
                dist.barrier()
                dist.broadcast(net.module.prototype_vectors, src=0)
                dist.barrier()
                torch.cuda.synchronize()
                # stage 3
                for j in range(10):
                    train(net, loader_train, optimizer_last_layer, criterion, scaler, args, local_rank,
                          stage=f'last_{j}', class_weight=class_weight)
    else:
        # NOET that the pushing results will be different from the original pushing (push before 10 epochs
        # last layer traing) !!! And the model will be changed accordingly !!! So we recommend using this 
        # code only for validation with our checkpoints in visualization.ipynb.
        
        # push_prototypes(
        #         loader_push,
        #         net.module,
        #         args,
        #         root_dir_for_saving_prototypes=img_dir if len(os.listdir(img_dir)) < 2 else None,
        #         prototype_img_filename_prefix=prototype_img_filename_prefix,
        #         proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix
        #         )
        
        pass
        
    # 9. evaluation
    del dataset_train, sampler_train, loader_train
    if local_rank == 0:
        f_x_i = np.zeros(y.shape)
        f_x_i[I_test], lcs_test, iads_test = test(net, loader_test, args, local_rank)
        f_x.append(f_x_i)
        del dataset_test, sampler_test, loader_test, dataset_push, sampler_push, loader_push
        for method, lcs_ in lcs_test.items():
            if not lcs.get(method):
                lcs[method] = {f'({a}, Th=0.5) {m}': np.zeros((cv_fold, 4))
                               for a in ['WT'] for m in ['AP', 'DSC']}
            for metric, lcs__ in lcs_.items():
                lcs[method][metric][cv_i] = lcs__.mean(0)
        if args.p_mode >= 0:
            n_prototypes_i = np.zeros((cv_fold, out_size))
            n_prototypes_i[cv_i] = net.module.prototype_class_identity.sum(0).cpu().numpy()
            n_prototype = n_prototypes_i[cv_i:cv_i + 1]
            n_prototypes.append(n_prototypes_i)
        else:
            n_prototype = None
        process_iad(iads_test, y[I_test], model_name=model_name_i)
        for method, iads_ in iads_test.items():
            if not iads.get(method):
                iads[method] = {m: np.zeros((cv_fold, 2)) for m in ['IA', 'ID', 'IAD']}
            for metric, iads__ in iads_.items():
                iads[method][metric][cv_i] = iads__
        print_results("Test", f_x_i[I_test], y[I_test], lcs_test, n_prototype, iads_test)
        if args.save_model and args.load_model is None:
            model_dir = './results/models/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net.module.state_dict(), f'{model_dir}{model_name_i}.pt')
    # 10. ddp destroy
    dist.destroy_process_group()
    torch.cuda.empty_cache()


def main():
    tic = time.time()
    # 1. parse arguments
    args = parse_arguments()
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
    # data augmentation (only training)
    if args.augmented:
        if args.aug_seq is None:
            args.aug_seq = 'af0-no0-bl0-br0-co0-an0-gi0-ga0-fl0'
        transform_aug = get_transform_aug(aug_seq=args.aug_seq)
    else:
        transform_aug = []
    transform_train = tio.Compose(transform_aug) if args.augmented else None
    transform_test = None
    del dataset, data_loader, transform, transform_aug
    toc = time.time()
    print(f"Elapsed time is {toc - tic:.6f} seconds.")

    # 4. 5-fold CV settings
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=args.seed)
    cv_fold = cv.get_n_splits()
    splits = np.zeros(y.shape[0], dtype=int)

    # 5. CV training
    args.p_mode = P_MODE_LIST[args.model_name]
    manager = mp.Manager()
    f_x = manager.list([])
    lcs = manager.dict({})
    n_prototypes = manager.list([])
    iads = manager.dict({})
    for i, (I_train, I_test) in enumerate(cv.split(x, y.argmax(1))):
        seed_everything(args.seed)
        print(f">>>>>>>> CV = {i + 1}:")
        splits[I_test] = i
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = os.environ['base_port']
        os.environ['WORLD_SIZE'] = os.environ['n_gpus']
        if int(os.environ['n_gpus']) == 1:
            train_one_fold(0, args, x, y, I_train, I_test, i, transform_train, transform_test, f_x, lcs, n_prototypes, iads, opts_hash)
        else:
            mp.spawn(
                train_one_fold,
                args=(args, x, y, I_train, I_test, i, transform_train, transform_test, f_x, lcs, n_prototypes, iads, opts_hash),
                nprocs=int(os.environ['n_gpus']),
                join=True,
            )
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
        print()
    f_x = np.sum(np.array(list(f_x)), axis=0)
    lcs = dict(lcs)
    n_prototypes = np.sum(np.array(list(n_prototypes)), axis=0)
    iads = dict(iads)
    
    # 6. overall evaluation
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
