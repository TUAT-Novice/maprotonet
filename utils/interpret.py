#!/usr/bin/env python3

import warnings
import captum.attr as ctattr
import torch.nn as nn

attr_methods: dict = {
    'P': 'ProtoNets',
    'D': 'Deconvolution',
    'G': 'GradCAM',
    'U': 'Guided GradCAM',
    'O': 'Occlusion',
}


def upsample(attr, data):
    if attr.ndim == 5:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:], interpolate_mode='trilinear')
    else:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:4], interpolate_mode='bilinear')
        attr = attr.reshape((data.shape[0], data.shape[4], 1) + data.shape[2:4]).permute(0, 2, 3, 4, 1)
    if attr.shape[0] // data.shape[0] == data.shape[1] // attr.shape[1]:
        return attr.reshape_as(data)
    else:
        return attr.expand_as(data).clone()


def attribute(net, data, target, method, show_progress=False):
    warnings.filterwarnings('ignore', message="Input Tensor \d+ did not already require gradients,")
    warnings.filterwarnings('ignore', message="Setting backward hooks on ReLU activations.The hook")
    warnings.filterwarnings('ignore', message="Setting forward, backward hooks and attributes on n")
    if method == 'ProtoNets':
        if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel):
            net = net.module
        if net.p_mode >= 1:
            # for XProtoNet, MProtoNet and MAProtoNet
            _, _, attr = net.push_forward(data)
        else:
            # for ProtoPNet
            _, attr = net.push_forward(data)
            attr = 1 - net.distance_2_similarity(attr)
        prototype_filters = net.prototype_class_identity[:, target].mT
        attr = (attr * prototype_filters[(...,) + (None,) * (attr.ndim - 2)]).mean(1, keepdim=True)
        return upsample(attr, data)
    elif method == 'Deconvolution':
        deconv = ctattr.Deconvolution(net)
        return deconv.attribute(data, target=target)
    elif method == 'GradCAM':
        conv_name = [n for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))][-1]
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        return upsample(attr, data)
    elif method == 'Guided GradCAM':
        conv_name = [n for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))][-1]
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        guided_bp = ctattr.GuidedBackprop(net)
        return guided_bp.attribute(data, target=target) * upsample(attr, data)
    elif method == 'Occlusion':
        occlusion = ctattr.Occlusion(net)
        sliding_window = (1,) + (11,) * len(data.shape[2:])
        strides = (1,) + (5,) * len(data.shape[2:])
        return occlusion.attribute(data, sliding_window, strides=strides, target=target, perturbations_per_eval=1, show_progress=show_progress)
