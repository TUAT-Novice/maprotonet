import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from functools import partial

from data.load import load_subjs_batch
from utils.interpret import attr_methods, attribute
from utils.metrics import lc, iad
from utils.utils import print_main


class FocalLoss(nn.modules.loss._WeightedLoss):
    __constants__ = ['gamma', 'ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(input, dim=0 if input.dim() == 1 else 1)
        return F.nll_loss((1 - log_prob.exp()) ** self.gamma * log_prob, target, weight=self.weight,
                          ignore_index=self.ignore_index, reduction=self.reduction)


def train(net, data_loader, optimizer, criterion, scaler, args, local_rank, **kwargs):
    assert args.use_amp is not None and scaler is not None and criterion is not None
    net.train()
    if args.p_mode >= 0:
        return train_ppnet(net, data_loader, optimizer, criterion, scaler, args, local_rank, **kwargs)
    for b, subjs_batch in enumerate(data_loader):
        data, target, _ = load_subjs_batch(subjs_batch)
        data = data.to(local_rank, non_blocking=True)
        target = target.argmax(1).to(local_rank, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            output = net(data)
            loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train_ppnet(net, data_loader, optimizer, criterion, scaler, args, local_rank, use_l1_mask=True, stage=None,
                log=print_main, class_weight=None):
    if stage in ['warm_up', 'joint']:
        log('', local_rank)
    log(f"\t{stage}", local_rank, end='', flush=True)
    start = time.time()
    n_examples, n_correct, n_batches = 0, 0, 0
    total_loss, total_cls, total_clst, total_sep, total_avg_sep, total_l1 = 0, 0, 0, 0, 0, 0
    total_map, total_oc = 0, 0
    for b, subjs_batch in enumerate(data_loader):
        data, target, _ = load_subjs_batch(subjs_batch)
        data = data.to(local_rank, non_blocking=True)
        target = target.argmax(1).to(local_rank, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            if args.p_mode >= 1:
                output, min_distances, h, p_map = net(data)
            else:
                output, min_distances = net(data)
            # Calculate classification loss
            classification = criterion(output, target)
            loss_cls = args.coefs['cls'] * classification
            total_cls += loss_cls.item()
            if stage in ['warm_up', 'joint']:
                # Calculate cluster loss
                max_dist = torch.prod(torch.tensor(net.module.prototype_shape[1:])).to(local_rank)
                target_weight = class_weight.to(local_rank)[target]
                target_weight = target_weight / target_weight.sum()
                prototypes_correct = net.module.prototype_class_identity[:, target].mT
                inv_distances_correct = ((max_dist - min_distances) * prototypes_correct).amax(1)
                cluster = ((max_dist - inv_distances_correct) * target_weight).sum()
                loss_clst = args.coefs['clst'] * cluster
                total_clst += loss_clst.item()
                # Calculate separation loss
                prototypes_wrong = 1 - prototypes_correct
                inv_distances_wrong = ((max_dist - min_distances) * prototypes_wrong).amax(1)
                separation = ((max_dist - inv_distances_wrong) * target_weight).sum()
                loss_sep = args.coefs['sep'] * separation
                total_sep += loss_sep.item()
                # Calculate average separation (only display)
                avg_separation = (min_distances * prototypes_wrong).sum(1) / prototypes_wrong.sum(1)
                avg_separation = (avg_separation * target_weight).sum()
                total_avg_sep += avg_separation.item()
                # Calculate mapping loss or multi-scale mapping loss
                if args.p_mode >= 3:
                    ri = torch.randint(2, (1,)).item()
                    f_affine = partial(
                        F.interpolate, scale_factor=(0.875, 0.75)[ri],
                        mode='trilinear' if h[-1].ndim == 5 else 'bilinear', align_corners=True
                    )
                    f_l1 = lambda t: t.abs().mean()
                    if args.p_mode >= 3:
                        h_A = [f_affine(h_i) for h_i in h]
                        h_A_mul = net.module.down_and_fuse(h_A)
                        mapping = f_l1(f_affine(p_map) - net.module.get_p_map(h_A_mul)) + f_l1(p_map)
                    else:
                        mapping = f_l1(f_affine(p_map) - net.module.get_p_map(f_affine(h[-1]))) + f_l1(p_map)
                    loss_map = args.coefs['map'] * mapping
                    total_map += loss_map.item()
                # Calculate online-CAM loss
                if args.p_mode >= 2:
                    h_ = net.module.down_and_fuse(h) if args.p_mode >= 3 else h[-1]
                    p_x = net.module.lse_pooling(net.module.p_map[:-3](h_).flatten(2))
                    output2 = net.module.last_layer(p_x @ net.module.p_map[-3].weight.flatten(1).mT)
                    online_cam = criterion(output2, target)
                    loss_oc = args.coefs['OC'] * online_cam
                    total_oc += loss_oc.item()
                # Calculate total loss
                loss = loss_cls + loss_clst + loss_sep
                if args.p_mode >= 1:
                    if args.p_mode <= 2 or args.mmloss:
                        loss = loss + loss_map
                if args.p_mode >= 2:
                    loss = loss + loss_oc
            else:  # for linear head
                # Calculate L1-regularization loss
                if use_l1_mask:
                    l1_mask = 1 - net.module.prototype_class_identity.mT
                    l1 = torch.linalg.vector_norm(net.module.last_layer.weight * l1_mask, ord=1)
                else:
                    l1 = torch.linalg.vector_norm(net.module.last_layer.weight, ord=1)
                loss_l1 = args.coefs['L1'] * l1
                total_l1 += loss_l1.item()
                # Calculate total loss
                loss = loss_cls + loss_l1
            total_loss += loss.item()
            # Evaluation statistics
            n_examples += target.shape[0]
            n_correct += (output.data.argmax(1) == target).sum().item()
            n_batches += 1
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    with torch.no_grad():
        p_avg_pdist = F.pdist(net.module.prototype_vectors.flatten(1)).mean().item()
    end = time.time()
    log(f"\ttime: {end - start:.2f}s,"
        f" acc: {n_correct / n_examples:.4f},"
        f" loss: {total_loss / n_batches:.4f},"
        f" cls: {total_cls / n_batches:.4f},"
        f" clst: {total_clst / n_batches:.4f},"
        f" sep: {total_sep / n_batches:.4f},"
        f" avg_sep: {total_avg_sep / n_batches:.4f},"
        f" L1: {total_l1 / n_batches:.4f},"
        f" map: {total_map / n_batches:.4f},"
        f" OC: {total_oc / n_batches:.4f},"
        f" p_avg_pdist: {p_avg_pdist:.4f}",
        local_rank)


def test(net, data_loader, args, local_rank):
    net.eval()
    f_x, lcs, iads = [], {}, {}
    if args.attr is not None:
        methods = args.attr
    elif args.p_mode >= 0:
        methods = 'P'  # ProtoPNets
    else:
        methods = 'G'  # GradCAM
    with torch.no_grad():
        for b, subjs_batch in enumerate(data_loader):
            data, target, seg_map = load_subjs_batch(subjs_batch)
            data = data.to(local_rank, non_blocking=True)
            target = target.argmax(1).to(local_rank, non_blocking=True)
            seg_map = seg_map.to(local_rank, non_blocking=True)
            f_x.append(F.softmax(net(data), dim=1).cpu().numpy())
            for method_i in methods:
                method = attr_methods[method_i]
                print_main(f" {method}:", local_rank, end='', flush=True)
                if not lcs.get(method):
                    lcs[method] = {f'({a}, Th=0.5) {m}': []
                                   for a in ['WT'] for m in ['AP', 'DSC']}
                if not iads.get(method):
                    iads[method] = {m: [] for m in ['IA', 'ID', 'IAD']}
                tic = time.time()
                attr = attribute(net, data, target, method)
                lcs[method]['(WT, Th=0.5) AP'].append(
                    lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='AP'))
                lcs[method]['(WT, Th=0.5) DSC'].append(
                    lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='DSC'))
                iads[method]['IA'].append(
                    iad(net, data, attr, n_intervals=50, quantile=True, addition=True))
                iads[method]['ID'].append(
                    iad(net, data, attr, n_intervals=50, quantile=True, addition=False))
                toc = time.time()
                print_main(f" {toc - tic:.6f}s,", local_rank, end='', flush=True)
            print_main(" Finished.", local_rank,)
    for method, lcs_ in lcs.items():
        for metric, lcs__ in lcs_.items():
            lcs[method][metric] = np.vstack(lcs__)
    for method, iads_ in iads.items():
        for metric, iads__ in iads_.items():
            if metric == 'IAD':
                continue
            iads[method][metric] = np.concatenate(iads__, axis=1)
    return np.vstack(f_x), lcs, iads
