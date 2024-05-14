# Implementation of data argumentation

# from: https://github.com/aywi/mprotonet/blob/master/src/utils.py

import torch
import torchio as tio
from functools import partial


def augment_brightness(t, multiplier_range=(0.5, 2)):
    multiplier = torch.empty(1).uniform_(multiplier_range[0], multiplier_range[1])
    return t * multiplier


def augment_contrast(t, contrast_range=(0.75, 1.25), preserve_range=True):
    if torch.rand(1) < 0.5 and contrast_range[0] < 1:
        factor = torch.empty(1).uniform_(contrast_range[0], 1)
    else:
        factor = torch.empty(1).uniform_(max(contrast_range[0], 1), contrast_range[1])
    t_mean = t.mean()
    if preserve_range:
        return ((t - t_mean) * factor + t_mean).clamp(t.min(), t.max())
    else:
        return (t - t_mean) * factor + t_mean


def augment_gamma(t, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-12, retain_stats=True):
    if invert_image:
        t = -t
    if retain_stats:
        t_mean, t_std = t.mean(), t.std()
    if torch.rand(1) < 0.5 and gamma_range[0] < 1:
        gamma = torch.empty(1).uniform_(gamma_range[0], 1)
    else:
        gamma = torch.empty(1).uniform_(max(gamma_range[0], 1), gamma_range[1])
    t_min = t.min()
    t_range = (t.max() - t_min).clamp_min(epsilon)
    t = ((t - t_min) / t_range) ** gamma * t_range + t_min
    if retain_stats:
        t = (t - t.mean()) / t.std().clamp_min(epsilon) * t_std + t_mean
    if invert_image:
        t = -t
    return t


transform_augs: dict = {
    'af0': tio.OneOf(
        {
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(0, 0), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.16,
            tio.RandomAffine(scales=(1, 1), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.16,
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.04,
        },
        p=0.36,
    ),
    'af1': tio.OneOf(
        {
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(0, 0), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
            tio.RandomAffine(scales=(1, 1), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
        },
        p=0.75,
    ),
    'mo1': tio.RandomMotion(p=0.1),
    'bi1': tio.RandomBiasField(coefficients=(0, 0.1), p=0.1),
    'no0': tio.RandomNoise(std=(0, 0.1), p=0.15),
    'bl0': tio.Compose(
        [
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t1']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t1ce']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t2']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['flair']),
        ],
        p=0.2,
    ),
    'br0': tio.Lambda(partial(augment_brightness, multiplier_range=(0.7, 1.3)),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'co0': tio.Lambda(partial(augment_contrast, contrast_range=(0.65, 1.5)),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'an0': tio.Compose(
        [
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t1']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t1ce']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t2']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['flair']),
        ],
        p=0.25,
    ),
    'gi0': tio.Lambda(partial(augment_gamma, gamma_range=(0.7, 1.5), invert_image=True),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'ga0': tio.Lambda(partial(augment_gamma, gamma_range=(0.7, 1.5), invert_image=False),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'fl0': tio.RandomFlip(axes=(0, 1, 2)),
    'fl1': tio.RandomFlip(),
}


def get_transform_aug(aug_seq):
    return [transform_augs[aug] for aug in aug_seq.split('-')]
