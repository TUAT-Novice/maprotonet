import os
import argparse
import warnings
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim

from thop import profile
from torch.utils.data import DataLoader

from utils.utils import seed_everything, print_main, print_param, print_results
from utils.metrics import process_iad
from utils.push import push_prototypes
from models.protos import ProtoNets, CNN
from train import FocalLoss, train, test


def train_one_fold(cv_i, opts_hash, local_rank=None, cv_fold=5):
    # 1. load temporary data and settings
    raw_data = torch.load('data.pt')
    x, y, transform_train, transform_test, args = \
        raw_data['x'], raw_data['y'], raw_data['transform_train'], raw_data['transform_test'], raw_data['args']
    idx = torch.load('index.pt')
    I_train, I_test = idx['I_train'], idx['I_test']
    metrics = torch.load('metrics.pt')
    f_x, lcs, n_prototypes, iads = metrics['f_x'], metrics['lcs'], metrics['n_prototypes'], metrics['iads']
    # settings
    torch.set_num_threads(args.n_threads)
    warnings.filterwarnings('ignore', message="The epoch parameter in `scheduler.step\(\)` was not")
    seed_everything(args.seed)
    if not args.use_da:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    local_rank = int(os.environ['LOCAL_RANK']) if local_rank is None else local_rank

    # 2. ddp initialize
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    torch.cuda.set_device(local_rank)

    # 3. dataset
    dataset_train = tio.SubjectsDataset(list(x[I_train]), transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.n_workers,
                              sampler=sampler_train, pin_memory=False, drop_last=True)
    if local_rank == 0:
        dataset_test = tio.SubjectsDataset(list(x[I_test]), transform=transform_test)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        loader_test = DataLoader(dataset_test, batch_size=args.batch_size // 2, num_workers=args.n_workers,
                                 sampler=sampler_test, pin_memory=False)
        if args.p_mode >= 0:
            dataset_push = tio.SubjectsDataset(list(x[I_train]), transform=transform_test)
            sampler_push = torch.utils.data.SequentialSampler(dataset_push)
            loader_push = DataLoader(dataset_push, batch_size=args.batch_size // 2, num_workers=args.n_workers,
                                     sampler=sampler_push, pin_memory=False)
    dist.barrier()

    # 4. model
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

    # 5. saving path
    img_dir = f'./results/saved_imgs/{model_name_i}/'
    if local_rank == 0 and not os.path.exists(img_dir):
        os.makedirs(img_dir)
    prototype_img_filename_prefix = 'prototype-img'
    proto_bound_boxes_filename_prefix = 'bb'

    # 6. optimizer
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

    # 7. warmup & lr scheduler
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

    # 8. balanced classification
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

    # 9. training
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    dist.barrier()
    if args.load_model is None:
        print_main("Epoch =", local_rank, end='', flush=True)
        for e in range(args.epoch):
            print_main(f" {e + 1}", local_rank, end='', flush=True)
            if args.lr_opt != 'Off':
                print_main(f"(lr={scheduler.get_last_lr()[0]:g})", local_rank, end='', flush=True)
            # stage 1
            train(net, loader_train, optimizer, criterion, scaler, args, local_rank, stage='joint',
                  class_weight=class_weight)
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
                dist.broadcast(net.module.prototype_vectors.clone().detach(), src=0)
                # stage 3
                for j in range(10):
                    train(net, loader_train, optimizer_last_layer, criterion, scaler, args, local_rank,
                          stage=f'last_{j}', class_weight=class_weight)

    # 10. evaluation
    del dataset_train, sampler_train, loader_train
    # push again for saving
    if local_rank == 0:
        push_prototypes(
            loader_push,
            net.module,
            args,
            root_dir_for_saving_prototypes=img_dir,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix
        )
        f_x[I_test], lcs_test, iads_test = test(net, loader_test, args, local_rank)
        del dataset_test, sampler_test, loader_test, dataset_push, sampler_push, loader_push
        for method, lcs_ in lcs_test.items():
            if not lcs.get(method):
                lcs[method] = {f'({a}, Th=0.5) {m}': np.zeros((cv_fold, 4))
                               for a in ['WT'] for m in ['AP', 'DSC']}
            for metric, lcs__ in lcs_.items():
                lcs[method][metric][cv_i] = lcs__.mean(0)
        if args.p_mode >= 0:
            n_prototypes[cv_i] = net.module.prototype_class_identity.sum(0).cpu().numpy()
            n_prototype = n_prototypes[cv_i:cv_i + 1]
        else:
            n_prototype = None
        process_iad(iads_test, y[I_test], model_name=model_name_i)
        for method, iads_ in iads_test.items():
            if not iads.get(method):
                iads[method] = {m: np.zeros((cv_fold, 2)) for m in ['IA', 'ID', 'IAD']}
            for metric, iads__ in iads_.items():
                iads[method][metric][cv_i] = iads__
        print_results("Test", f_x[I_test], y[I_test], lcs_test, n_prototype, iads_test)
        if args.save_model and args.load_model is None:
            model_dir = './results/models/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net.module.state_dict(), f'{model_dir}{model_name_i}.pt')

        # 11. save temporary data
        torch.save({'f_x':  f_x, 'lcs': lcs, 'n_prototypes': n_prototypes, 'iads': iads}, 'metrics.pt')


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-i', type=int, required=True, help="index of cv fold")
    parser.add_argument('--opts-hash', type=str, required=True, help="hash code for saving")
    parser.add_argument('--local-rank', type=int, default=None, help="local rank for ddp")
    parser.add_argument('--cv_fold', type=int, default=5, help="number of total cv fold")
    args = parser.parse_args()
    train_one_fold(args.cv_i, args.opts_hash, args.local_rank, cv_fold=args.cv_fold)
    dist.destroy_process_group()
    torch.cuda.empty_cache()

