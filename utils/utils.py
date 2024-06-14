import random
import numpy as np
import torch

import socket
import joblib
import time
import zlib

from utils.metrics import accuracy, balanced_accuracy, sensitivity, specificity, auroc


# seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_main(t, local_rank):
    if local_rank == 0:
        print(t)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def f_splits(splits, f_metric, f_x, y):
    indexes = np.unique(splits)
    metrics = np.zeros(indexes.shape)
    for i, index in enumerate(indexes):
        metrics[i] = f_metric(f_x[splits == index], y[splits == index])
    return metrics


# 1. print
def print_param(net, show_each=True):
    max_len_name = 0
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            if show_each:
                if len(name) > max_len_name:
                    max_len_name = len(name)
            n_param += np.prod(param.shape)
    if show_each:
        for name, param in net.named_parameters():
            if param.requires_grad:
                print("{:{:d}s}    {:s}".format(name, max_len_name, str(param.shape)))
    print(f"Number of Parameters = {n_param}")


def print_results(dataset, f_x, y, lcs=None, n_prototypes=None, iads=None, splits=None):
    print(f"{dataset}", end='', flush=True)
    if splits is not None:
        accs = f_splits(splits, accuracy, f_x, y)
        bacs = f_splits(splits, balanced_accuracy, f_x, y)
        sens = f_splits(splits, sensitivity, f_x, y)
        spes = f_splits(splits, specificity, f_x, y)
        aucs = f_splits(splits, auroc, f_x, y)
        print(f" ACC: {accs.mean():.3f}±{accs.std():.3f},"
              f" BAC: {bacs.mean():.3f}±{bacs.std():.3f},"
              f" SEN: {sens.mean():.3f}±{sens.std():.3f},"
              f" SPE: {spes.mean():.3f}±{spes.std():.3f},"
              f" AUC: {aucs.mean():.3f}±{aucs.std():.3f}")
    else:
        print(f" ACC: {accuracy(f_x, y):.3f},"
              f" BAC: {balanced_accuracy(f_x, y):.3f},"
              f" SEN: {sensitivity(f_x, y):.3f},"
              f" SPE: {specificity(f_x, y):.3f},"
              f" AUC: {auroc(f_x, y):.3f}")
    if lcs:
        maxlen_method, maxlen_metric = 0, 0
        for method, lcs_ in lcs.items():
            for metric, lcs__ in lcs_.items():
                maxlen_method = max(len(method), maxlen_method)
                maxlen_metric = max(len(metric), maxlen_metric)
        for method, lcs_ in lcs.items():
            for metric, lcs__ in lcs_.items():
                print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                      f" {lcs__.mean(1).mean():.3f}±{lcs__.mean(1).std():.3f}"
                      f" (T1: {lcs__[:, 0].mean():.3f}±{lcs__[:, 0].std():.3f},"
                      f" T1CE: {lcs__[:, 1].mean():.3f}±{lcs__[:, 1].std():.3f},"
                      f" T2: {lcs__[:, 2].mean():.3f}±{lcs__[:, 2].std():.3f},"
                      f" FLAIR: {lcs__[:, 3].mean():.3f}±{lcs__[:, 3].std():.3f})")
    if n_prototypes is not None:
        for n_prototype in n_prototypes:
            print(f"        No. of Prototypes\t"
                  f" All: {n_prototype.sum():.0f},"
                  f" HGG: {n_prototype[1]:.0f},"
                  f" LGG: {n_prototype[0]:.0f}")
        if len(n_prototypes) > 1:
            print(f"Average No. of Prototypes\t"
                  f" All: {n_prototypes.sum(1).mean():.1f}±{n_prototypes.sum(1).std():.1f},"
                  f" HGG: {n_prototypes[:, 1].mean():.1f}±{n_prototypes[:, 1].std():.1f},"
                  f" LGG: {n_prototypes[:, 0].mean():.1f}±{n_prototypes[:, 0].std():.1f}")
    if iads:
        maxlen_method, maxlen_metric = 0, 0
        for method, iads_ in iads.items():
            for metric, iads__ in iads_.items():
                maxlen_method = max(len(method), maxlen_method)
                maxlen_metric = max(len(metric), maxlen_metric)
        for method, iads_ in iads.items():
            for metric, iads__ in iads_.items():
                print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                      f" {iads__[:, 0].mean():.3f}±{iads__[:, 0].std():.3f} (ACC),"
                      f" {iads__[:, 1].mean():.3f}±{iads__[:, 1].std():.3f} (BAC)")


def get_hashes(args):
    args_ignored = ['n_workers', 'n_threads', 'local_rank', 'gpus', 'device', 'device_id', 'load_model', 'save_model']
    opts = {k: v for k, v in vars(args).items() if k not in args_ignored}
    opts_hash = f"{zlib.crc32(str(opts).encode()):x}"
    return opts_hash


def output_results(dataset, args, f_x, y, lcs=None, n_prototypes=None,
                   iads=None, splits=None, file='./results/raw.md'):
    hostname = socket.gethostname()
    # https://stackoverflow.com/a/28950776
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(0)
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    cur_time = time.strftime('%Y/%m/%d %H:%M:%S %z')
    opts_hash = get_hashes(args)
    with open(file, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f"# {hostname} ({ip}): {cur_time}\n\n")
        f.write(f"```python\n{args}\n```\n\n")
        f.write(f"## Results: {dataset}\n\n")
        f.write("|Model|BS|ACC|BAC|SEN|SPE|AUC|\n")
        f.write("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
        f.write(f"|{args.model_name}_{opts_hash}|{args.batch_size}|")
        if splits is not None:
            accs = f_splits(splits, accuracy, f_x, y)
            bacs = f_splits(splits, balanced_accuracy, f_x, y)
            sens = f_splits(splits, sensitivity, f_x, y)
            spes = f_splits(splits, specificity, f_x, y)
            aucs = f_splits(splits, auroc, f_x, y)
            f.write(f"{accs.mean():.3f}±{accs.std():.3f}|"
                    f"{bacs.mean():.3f}±{bacs.std():.3f}|"
                    f"{sens.mean():.3f}±{sens.std():.3f}|"
                    f"{spes.mean():.3f}±{spes.std():.3f}|"
                    f"{aucs.mean():.3f}±{aucs.std():.3f}|\n\n")
        else:
            f.write(f"{accuracy(f_x, y):.3f}|"
                    f"{balanced_accuracy(f_x, y):.3f}|"
                    f"{sensitivity(f_x, y):.3f}|"
                    f"{specificity(f_x, y):.3f}|"
                    f"{auroc(f_x, y):.3f}|\n\n")
        if lcs:
            f.write(f"|Method|Metric|Average|T1|T1CE|T2|FLAIR|\n")
            f.write("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
            for method, lcs_ in lcs.items():
                for metric, lcs__ in lcs_.items():
                    f.write(f"|{method}|{metric}|"
                            f"{lcs__.mean(1).mean():.3f}±{lcs__.mean(1).std():.3f}|"
                            f"{lcs__[:, 0].mean():.3f}±{lcs__[:, 0].std():.3f}|"
                            f"{lcs__[:, 1].mean():.3f}±{lcs__[:, 1].std():.3f}|"
                            f"{lcs__[:, 2].mean():.3f}±{lcs__[:, 2].std():.3f}|"
                            f"{lcs__[:, 3].mean():.3f}±{lcs__[:, 3].std():.3f}|\n")
            f.write('\n')
        if n_prototypes is not None:
            f.write(f"|CV|All|HGG|LGG|\n")
            f.write("|:-:|:-:|:-:|:-:|\n")
            for cv, n_prototype in enumerate(n_prototypes):
                f.write(f"|{cv + 1}|"
                        f"{n_prototype.sum():.0f}|"
                        f"{n_prototype[1]:.0f}|"
                        f"{n_prototype[0]:.0f}|\n")
            f.write(f"|Average|"
                    f"{n_prototypes.sum(1).mean():.1f}±{n_prototypes.sum(1).std():.1f}|"
                    f"{n_prototypes[:, 1].mean():.1f}±{n_prototypes[:, 1].std():.1f}|"
                    f"{n_prototypes[:, 0].mean():.1f}±{n_prototypes[:, 0].std():.1f}|\n")
            f.write('\n')
        if iads:
            f.write(f"|Method|Metric|ACC|BAC|\n")
            f.write("|:-:|:-:|:-:|:-:|\n")
            for method, iads_ in iads.items():
                for metric, iads__ in iads_.items():
                    f.write(f"|{method}|{metric}|"
                            f"{iads__[:, 0].mean():.3f}±{iads__[:, 0].std():.3f}|"
                            f"{iads__[:, 1].mean():.3f}±{iads__[:, 1].std():.3f}|\n")
            f.write('\n')


def save_cvs(cv_dir, args, f_x, y, lcs, iads, splits):
    opts_hash = get_hashes(args)
    file = f'{cv_dir}{args.model_name}_{opts_hash}.joblib'
    bacs = f_splits(splits, balanced_accuracy, f_x, y)
    aucs = f_splits(splits, auroc, f_x, y)
    method = 'ProtoNets' if lcs.get('ProtoNets') else 'GradCAM'
    aps = lcs[method]['(WT, Th=0.5) AP'].mean(1)
    dscs = lcs[method]['(WT, Th=0.5) DSC'].mean(1)
    method = 'ProtoNets' if iads.get('ProtoNets') else 'GradCAM'
    ias = iads[method]['IA'][:, 1]
    ids = iads[method]['ID'][:, 1]
    cvs = {'bacs': bacs, 'aucs': aucs, 'aps': aps, 'dscs': dscs, 'ias': ias, 'ids': ids}
    joblib.dump(cvs, file, compress=True)


# https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
# https://stackoverflow.com/q/36901
# https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions
def replace_module(net, type_old, type_new, *args, **kwargs):
    if isinstance(type_old, list):
        assert len(type_old) == 2
        type_old_, attrs = type_old
        assert isinstance(attrs, dict)
        for n, m in net.named_children():
            if isinstance(m, type_old_):
                matched = True
                for attr, value in attrs.items():
                    if getattr(m, attr) != value:
                        matched = False
                        break
                if matched:
                    setattr(net, n, type_new(*args, **kwargs))
            else:
                replace_module(m, type_old, type_new, *args, **kwargs)
    else:
        for n, m in net.named_children():
            if isinstance(m, type_old):
                setattr(net, n, type_new(*args, **kwargs))
            else:
                replace_module(m, type_old, type_new, *args, **kwargs)
