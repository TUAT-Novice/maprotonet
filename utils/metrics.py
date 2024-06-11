import torch
import numpy as np
import os
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score)
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F


def accuracy(f_x, y):
    return accuracy_score(y.argmax(1), f_x.argmax(1))


def balanced_accuracy(f_x, y):
    return balanced_accuracy_score(y.argmax(1), f_x.argmax(1))


def sensitivity(f_x, y):
    return recall_score(y.argmax(1), f_x.argmax(1), pos_label=1)


def specificity(f_x, y):
    return recall_score(y.argmax(1), f_x.argmax(1), pos_label=0)


def auroc(f_x, y):
    return roc_auc_score(y, f_x)


def lc(prob, seg, annos=None, threshold=None, topk_v=None, metric='AP'):
    lc_is = []
    for prob_i, seg_i in zip(prob, seg):
        lc_ijs = []
        anno = torch.zeros_like(seg_i[0])
        if isinstance(annos, list):
            for a in annos:
                anno[seg_i[0] == a] = 1
        else:
            anno[seg_i[0] > 0] = 1
        if threshold is None:
            if topk_v:
                quantile = 1 - topk_v / 100
            else:
                quantile = 1 - anno.sum() / anno.numel()
            threshold = torch.quantile(prob_i, quantile)
        else:
            prob_i = prob_i - prob_i.min()
            prob_i = prob_i / prob_i.max().clamp_min(1e-12)
        for prob_ij in prob_i:
            attr = torch.ones_like(prob_ij)
            attr[prob_ij < threshold] = 0
            if metric == 'AP':
                lc_ijs.append((anno * attr).sum() / attr.sum().clamp_min(1))
            elif metric == 'DSC':
                lc_ijs.append(2 * (anno * attr).sum() / (anno.sum() + attr.sum()).clamp_min(1))
            elif metric == 'IoU':
                lc_ijs.append((anno * attr).sum() / torch.logical_or(anno, attr).sum().clamp_min(1))
        lc_is.append(torch.hstack(lc_ijs))
    return torch.vstack(lc_is).cpu().numpy()


# Incremental addition/deletion
def iad(net, data, attr, n_intervals=100, uniform=True, quantile=True, addition=True):
    iads = []
    quantiles = torch.linspace(1, 0, n_intervals + 1).to(attr.device)
    if uniform:
        attr = attr.mean(1, keepdim=True)
    if quantile:
        thresholds = torch.quantile(attr.flatten(1), quantiles, dim=1)
    else:
        attr = attr - attr.amin(tuple(range(1, attr.ndim)), keepdim=True)
        attr = attr / attr.amax(tuple(range(1, attr.ndim)), keepdim=True).clamp_min(1e-12)
        thresholds = quantiles[:, None].tile(1, attr.shape[0])
    for i, threshold in enumerate(thresholds):
        if addition:
            mask = torch.zeros_like(attr)
            if i > 0:
                mask[attr >= threshold[(...,) + (None,) * (attr.ndim - 1)]] = 1
        else:
            mask = torch.ones_like(attr)
            if i > 0:
                mask[attr >= threshold[(...,) + (None,) * (attr.ndim - 1)]] = 0
        with torch.no_grad():
            iads.append(F.softmax(net(data * mask), dim=1))
    return torch.stack(iads).cpu().numpy()


def plot_iad(curve, ratio, method, metric, x_label, y_label, show=True, save=False, path=None):
    metric = {'IA': "Incremental Addition", 'ID': "Incremental Deletion"}[metric]
    x0 = np.linspace(0, 100, curve.shape[0])
    x = np.linspace(0, 100, (curve.shape[0] - 1) * 10 + 1)
    curve = np.interp(x, x0, curve)
    bounds = sorted([(curve[0], 'Start'), (curve[-1], 'End')])
    if show:
        # https://stackoverflow.com/a/56320309
        matplotlib.use('qt5agg')
    else:
        matplotlib.use('agg')
    plt.figure(figsize=(6, 4))
    h_c, = plt.plot(x, curve, label=metric)
    h_r = plt.fill_between(x, curve.clip(bounds[0][0], bounds[1][0]), bounds[0][0],
                           where=curve > bounds[0][0], color='dodgerblue', alpha=0.3,
                           label=f"Score: {ratio:.3f}")
    h_l = plt.axhline(bounds[0][0], color='r', alpha=0.7, linestyle=':',
                      label=f"Lower Bound ({bounds[0][1]})")
    h_u = plt.axhline(bounds[1][0], color='r', alpha=0.7, linestyle=(0, (2, 3)),
                      label=f"Upper Bound ({bounds[1][1]})")
    plt.legend(handles=[h_c, h_u, h_r, h_l])
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if path[-3:] != 'pdf':
        plt.title(f"{method}: {metric} ({y_label})")
    if save and path is not None:
        plt.savefig(path, dpi=300, pad_inches=0)
        # print(f"Output {path}")
    if show:
        plt.show()
    else:
        plt.close()


def process_iad(iads, y, save_plot=True, model_name=None):
    for method, iads_ in iads.items():
        for metric, iads__ in iads_.items():
            if metric == 'IAD':
                continue
            curves = np.zeros((iads__.shape[0], 2))
            for i in range(iads__.shape[0]):
                curves[i, 0] = accuracy(iads__[i], y)
                curves[i, 1] = balanced_accuracy(iads__[i], y)
            ratios = curves.clip(curves[[0, -1], :].min(0), curves[[0, -1], :].max(0))
            ratios = ratios - ratios.min(0)
            ratios = ratios / ratios.max(0).clip(1e-12)
            iads[method][metric] = ((ratios[:-1] + ratios[1:]) / 2).mean(0, keepdims=True)
            if save_plot and model_name is not None:
                iads_dir = f'./results/iads/{model_name}/'
                if not os.path.exists(iads_dir):
                    os.makedirs(iads_dir)
                plot_iad(curves[:, 0], iads[method][metric][0, 0], method, metric, "Top Voxels (%)",
                         "Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_ACC.pdf')
                plot_iad(curves[:, 0], iads[method][metric][0, 0], method, metric, "Top Voxels (%)",
                         "Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_ACC.png')
                plot_iad(curves[:, 1], iads[method][metric][0, 1], method, metric, "Top Voxels (%)",
                         "Balanced Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_BAC.pdf')
                plot_iad(curves[:, 1], iads[method][metric][0, 1], method, metric, "Top Voxels (%)",
                         "Balanced Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_BAC.png')
        if 'IAD' in iads_:
            iads[method]['IAD'] = (iads[method]['IA'] + (1 - iads[method]['ID'])) / 2
