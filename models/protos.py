# Implementation of prototypical part networks

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.receptive_field import compute_proto_layer_rf_info_v2
from models.resnet3d import ResidualNet
from utils.utils import replace_module


def build_resnet_backbone(net):
    return nn.Sequential(OrderedDict([
        ('conv1', net.conv1),
        ('bn1', net.bn1),
        ('relu', net.relu),
        ('maxpool', net.maxpool),
        ('layer1', net.layer1),
        ('layer2', net.layer2),
        ('layer3', net.layer3),
        ('layer4', net.layer4)
    ]))


def resnet_imagenet(backbone):
    # resnet
    if backbone == 'resnet18':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=18, num_classes=2, att_type=None))
    elif backbone == 'resnet34':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=34, num_classes=2, att_type=None))
    elif backbone == 'resnet50':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=50, num_classes=2, att_type=None))
    elif backbone == 'resnet101':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=101, num_classes=2, att_type=None))
    elif backbone == 'resnet152':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=152, num_classes=2, att_type=None))
    # resnet + triplet attention
    elif backbone == 'resnet18_quad':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=18, num_classes=2, att_type="QuadrupletAttention"))
    elif backbone == 'resnet34_quad':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=34, num_classes=2, att_type="QuadrupletAttention"))
    elif backbone == 'resnet50_quad':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=50, num_classes=2, att_type="QuadrupletAttention"))
    elif backbone == 'resnet101_quad':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=101, num_classes=2, att_type="QuadrupletAttention"))
    elif backbone == 'resnet152_quad':
        return build_resnet_backbone(ResidualNet("ImageNet", depth=152, num_classes=2, att_type="QuadrupletAttention"))


def init_conv(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_resnet3d(net, in_channels=3):
    # adjust input channel
    replace_module(net, [nn.Conv2d, {'in_channels': 3, 'out_channels': 64}], nn.Conv3d, in_channels,
                   64, 7, stride=2, padding=3, bias=False)
    replace_module(net, [nn.Conv3d, {'in_channels': 3, 'out_channels': 64}], nn.Conv3d, in_channels,
                   64, 7, stride=2, padding=3, bias=False)
    # double check for 3d block
    for ic, oc, ks, st, pd in [(64, 64, 1, 1, 0), (64, 64, 3, 1, 1), (64, 128, 1, 2, 0),
                               (64, 128, 3, 2, 1), (64, 256, 1, 1, 0), (128, 128, 3, 1, 1),
                               (128, 128, 3, 2, 1), (128, 256, 1, 2, 0), (128, 256, 3, 2, 1),
                               (128, 512, 1, 1, 0), (256, 64, 1, 1, 0), (256, 128, 1, 1, 0),
                               (256, 256, 3, 1, 1), (256, 256, 3, 2, 1), (256, 512, 1, 2, 0),
                               (256, 512, 3, 2, 1), (256, 1024, 1, 1, 0), (512, 128, 1, 1, 0),
                               (512, 256, 1, 1, 0), (512, 512, 3, 1, 1), (512, 512, 3, 2, 1),
                               (512, 1024, 1, 2, 0), (512, 2048, 1, 1, 0), (1024, 256, 1, 1, 0),
                               (1024, 512, 1, 1, 0), (1024, 2048, 1, 2, 0), (2048, 512, 1, 1, 0)]:
        replace_module(net,
                       [nn.Conv2d, {'in_channels': ic, 'out_channels': oc, 'kernel_size': (ks, ks),
                                    'stride': (st, st), 'padding': (pd, pd)}],
                       nn.Conv3d, ic, oc, ks, stride=st, padding=pd, bias=False)
    for nf in [64, 128, 256, 512, 1024, 2048]:
        replace_module(net, [nn.BatchNorm2d, {'num_features': nf}], nn.BatchNorm3d, nf)
    replace_module(net, nn.MaxPool2d, nn.MaxPool3d, kernel_size=3, stride=2, padding=1)
    # initialize
    init_conv(net)


def conv_info(net):
    kernel_sizes, strides, paddings = [], [], []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if m.kernel_size[0] == 1 and m.stride[0] == 2:
                continue
            kernel_sizes += [m.kernel_size[0]]
            strides += [m.stride[0]]
            paddings += [m.padding[0]]
        elif isinstance(m, (nn.MaxPool2d, nn.MaxPool3d)):
            kernel_sizes += [m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]]
            strides += [m.stride if isinstance(m.stride, int) else m.stride[0]]
            paddings += [m.padding if isinstance(m.padding, int) else m.padding[0]]
    return kernel_sizes, strides, paddings


# ------------------------------------------- Resnet3D -------------------------------------------
class CNN(nn.Module):
    def __init__(self, in_size=(4, 128, 128, 96), out_size=2, backbone='resnet152', n_res_block=2, **kwargs):
        super(CNN, self).__init__()
        # 1. backbone
        # +4 denotes the input block: conv1, bn1, relu1 and maxpool
        self.backbone = resnet_imagenet(backbone)[:n_res_block + 4]
        self.backbone_name = backbone + f'[:{n_res_block}]' if n_res_block else backbone
        init_resnet3d(self.backbone, in_channels=in_size[0])
        # 2. features add ons
        add_ons_channels = [64, 256, 512, 1024][:n_res_block + 1][-1]
        self.features_add_ons = nn.Sequential(
            nn.Conv3d(add_ons_channels, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1)
        )
        init_conv(self.features_add_ons)
        # 3. linear head
        self.fc = nn.Linear(128, out_size, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.features_add_ons(x)
        x = self.fc(x)
        return x


# -------------------------------- Prototypical Part Network 3D --------------------------------
class PoolingGate(nn.Module):
    def __init__(self, downsample=2):
        super(PoolingGate, self).__init__()
        self.avg = nn.AvgPool3d(kernel_size=downsample+1, stride=downsample, padding=downsample//2)
        self.max = nn.MaxPool3d(kernel_size=downsample+1, stride=downsample, padding=downsample//2)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        return torch.cat([avg, max], dim=1)


class ProtoNets(nn.Module):
    def __init__(self,
                 in_size=(4, 128, 128, 96),
                 out_size=2,
                 backbone='resnet152_quadruplet',
                 n_res_block=2,
                 prototype_shape=(30, 128, 1, 1, 1),
                 f_dist='cos',
                 prototype_activation_function='log',
                 p_mode=3,  # ** {0: ProtoPNet, 1: XProtoNet, 2: MProtoNet, 3: MAProtoNet} **
                 topk_p=1,  # only for ProtoPNet
                 n_scales=2,  # only for MAProtoNet
                 scale_mode='c',  # only for MAProtoNet
                 **kwargs):
        super(ProtoNets, self).__init__()
        self.in_size = in_size
        self.num_classes = out_size
        self.p_mode = p_mode
        self.n_res_block = n_res_block
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.f_dist = f_dist
        self.topk_p = topk_p
        self.n_scales = n_scales
        self.scale_mode = scale_mode
        self.epsilon = 1e-4
        # prototype_activation_function could be 'log', 'linear', or a generic function that
        # converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function
        # Do not make this just a tensor, since it will not be automatically moved to GPU(s)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # 0. learnable prototype vectors v
        assert self.num_prototypes % self.num_classes == 0
        # A onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = nn.Parameter(
            torch.zeros(self.num_prototypes, self.num_classes), requires_grad=False)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        while len(self.prototype_shape) < 5:
            self.prototype_shape += (1,)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        # 1. backbone H(x)
        # +4 denotes the input block: conv1, bn1, relu1 and maxpool
        self.backbone = resnet_imagenet(backbone)[:n_res_block + 4]
        self.backbone_name = backbone + f'[:{n_res_block}]' if n_res_block else backbone
        # receptive field information
        layer_filter_sizes, layer_strides, layer_paddings = conv_info(self.backbone)
        self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=self.in_size[1], layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides, layer_paddings=layer_paddings,
            prototype_kernel_size=self.prototype_shape[2]
        )

        # 2. features_add_ons F(x)
        # +1 denotes the raw features before resnet block (after (conv1, bn1, relu1 and maxpool)) are also considered
        add_ons_channels = [64, 256, 512, 1024][:n_res_block + 1][-1]
        self.features_add_ons = nn.Sequential(
            nn.Conv3d(add_ons_channels, self.prototype_shape[1], 1, bias=False),
            nn.BatchNorm3d(self.prototype_shape[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.prototype_shape[1], self.prototype_shape[1], 1, bias=False),
            nn.BatchNorm3d(self.prototype_shape[1]),
            nn.Sigmoid()
        )

        # 3. mapping module M(x)
        # for ProtoPNet, not mapping module
        if self.p_mode == 0:
            # ProtoPNet
            p_size_w = compute_proto_layer_rf_info_v2(
                img_size=self.in_size[2], layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides, layer_paddings=layer_paddings,
                prototype_kernel_size=self.prototype_shape[3]
            )[0]
            p_size_a = compute_proto_layer_rf_info_v2(
                img_size=self.in_size[3], layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides, layer_paddings=layer_paddings,
                prototype_kernel_size=self.prototype_shape[4]
            )[0]
            self.p_size = (self.proto_layer_rf_info[0], p_size_w, p_size_a)
            self.topk_p = int(self.p_size[0] * self.p_size[1] * self.p_size[2] * topk_p / 100)
            assert self.topk_p >= 1
        # for XProtoNet and MProtoNet, a single-scale mapping module
        elif self.p_mode in [1, 2]:
            # for M(x) of XProtoNet, MProtoNet
            self.p_map = nn.Sequential(
                nn.Conv3d(add_ons_channels, self.prototype_shape[1], 1, bias=False),
                nn.BatchNorm3d(self.prototype_shape[1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.prototype_shape[1], self.prototype_shape[0], 1, bias=False),
                nn.BatchNorm3d(self.prototype_shape[0]),
                nn.Sigmoid()
            )
        # for our MAProtoNet, a multi-scale mapping module
        elif self.p_mode >= 3:
            assert n_scales >= 1, f"Expected n_scales >= 1, but get {n_scales}"
            assert n_res_block + 1 >= n_scales, f"You have scale-level={n_res_block + 1}, but set n_scales={n_scales}"
            in_chans = [64, 256, 512, 1024][:n_res_block + 1][::-1][:n_scales]
            # 3.1. the pre-multi-scale module MS(x)
            if scale_mode in ['a', 'b']:  # down-sampling through convolution
                out_chans = add_ons_channels * n_scales if scale_mode == 'a' else add_ons_channels
                self.scale_layers = [
                    [
                        nn.Conv3d(in_chans[i], add_ons_channels, kernel_size=3, stride=2 * i, padding=i, dilation=i),
                        nn.BatchNorm3d(add_ons_channels),
                        nn.ReLU()
                    ]
                    for i in range(1, n_scales)
                ]
                self.scale_layers = self.scale_layers[::-1]
            elif scale_mode == 'c':  # down-sampling through pooling
                out_chans = add_ons_channels
                for i in range(1, n_scales):
                    out_chans += 2 * in_chans[i]
                self.scale_layers = [
                    [
                        PoolingGate(downsample=2*i),
                    ]
                    for i in range(1, n_scales)
                ]
            elif scale_mode == 'd':  # down-sampling through pooling
                out_chans = add_ons_channels
                self.scale_layers = [
                    [
                        PoolingGate(downsample=2*i),
                        nn.Conv3d(in_chans[i] * 2, add_ons_channels, kernel_size=1),
                        nn.BatchNorm3d(add_ons_channels),
                        nn.ReLU()
                    ]
                    for i in range(1, n_scales)
                ]
            self.scale_layers = nn.Sequential(
                *[nn.Sequential(*s) for s in self.scale_layers]
            )
            # 3.2. mapping module M(x) especially for our MAProtoNet
            self.p_map = nn.Sequential(
                nn.Conv3d(out_chans, self.prototype_shape[1], 1, bias=False), # concat
                nn.BatchNorm3d(self.prototype_shape[1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.prototype_shape[1], self.prototype_shape[0], 1, bias=False),
                nn.BatchNorm3d(self.prototype_shape[0]),
                nn.Sigmoid()
            )

        # 4. linear head C(x)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self._initialize_weights()

    def scale(self, x, dim):
        x = x - x.amin(dim, keepdim=True)
        return x / x.amax(dim, keepdim=True).clamp_min(self.epsilon)

    def sigmoid(self, x, omega=10, sigma=0.5):
        return torch.sigmoid(omega * (x - sigma))

    def lse_pooling(self, x, r=10, dim=-1):
        return (torch.logsumexp(r * x, dim=dim) - torch.log(torch.tensor(x.shape[dim]))) / r

    def get_multiscale_h(self, x):
        res1 = self.backbone[:3](x)
        outs = (res1,)
        if self.n_res_block >= 1:
            outs += (self.backbone[3:5](outs[-1]),)
        if self.n_res_block >= 2:
            outs += (self.backbone[5](outs[-1]),)
        if self.n_res_block >= 3:
            outs += (self.backbone[6](outs[-1]),)
        if self.n_res_block >= 4:
            outs += (self.backbone[7](outs[-1]),)
        return outs

    def down_and_fuse(self, x):
        assert len(x) >= self.n_scales, \
            f'No enough scale to unpack, {len(x)} scales are given but {self.n_scales} scales are required'
        x = x[-self.n_scales:]
        # down-sampling
        outs = []
        for scale_i in range(self.n_scales - 1):
            outs.append(self.scale_layers[scale_i](x[scale_i]))
        outs.append(x[-1])
        # fusion
        if self.scale_mode in ['a', 'c']:
            multiscale_outs = torch.cat(outs, dim=1)  # concat
        elif self.scale_mode in ['b', 'd']:
            multiscale_outs = sum(outs)  # add
        return multiscale_outs

    def get_p_map(self, x):
        if self.p_mode >= 2:  # for MProtoNet and MAProtoNet
            p_map = F.relu(self.p_map[:-1](x))
            p_map = self.scale(p_map, tuple(range(1, p_map.ndim)))  # soft masking
            return self.sigmoid(p_map)
        else:  # for XProtoNet
            return self.p_map(x)

    def conv_features(self, x):
        h = self.get_multiscale_h(x)
        f_x = self.features_add_ons(h[-1])
        if self.p_mode == 0:  # for ProtoPNet
            return f_x
        elif self.p_mode in [1, 2]:  # for XProtoNet and MProtoNet
            p_map = self.get_p_map(h[-1])  # single-scale features
            return f_x, p_map, h
        elif self.p_mode >= 3:  # for our MAProtoNet
            p_map = self.get_p_map(self.down_and_fuse(h))  # multi-scale features
            return f_x, p_map, h

    def l2_convolution_3D(self, x):
        if x.shape[1:] == self.prototype_shape:
            x_2 = (x ** 2).sum(2)
            xp = (x * self.prototype_vectors).sum(2)
        else:
            x_2 = F.conv3d(x ** 2, self.ones)
            xp = F.conv3d(x, self.prototype_vectors)
        p_2 = (self.prototype_vectors ** 2).sum((1, 2, 3, 4)).reshape(-1, 1, 1, 1)
        return F.relu(x_2 - 2 * xp + p_2)

    def cosine_convolution_3D(self, x):
        assert x.min() >= 0, f"{x.min():.16g} >= 0"
        prototype_vectors_unit = F.normalize(self.prototype_vectors, p=2, dim=1)
        if x.shape[1:] == self.prototype_shape:
            x_unit = F.normalize(x, p=2, dim=2)
            return F.relu(1 - (x_unit * prototype_vectors_unit).sum(2))
        else:
            x_unit = F.normalize(x, p=2, dim=1)
            return F.relu(1 - F.conv3d(input=x_unit, weight=prototype_vectors_unit))

    def prototype_distances(self, f_x, p_map=None):
        # calculate distance
        if self.p_mode == 0:  # for ProtoPNet, calculate directly through features
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(f_x)
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(f_x)
        if self.p_mode >= 1: # for XProtoNet, MProtoNet and MAProtoNet, calculate after masking
            assert p_map is not None, "Except 'p_map' for distance calculation, but received 'None'"
            p_size = f_x.flatten(2).shape[2]
            p_x = (torch.einsum('bphwa,bchwa->bpc', p_map, f_x) / p_size)[(...,) + (None,) * 3]
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(p_x)
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(p_x)

    def distance_2_similarity(self, distances):
        if self.f_dist == 'cos':
            return F.relu(1 - distances)
        elif self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        if self.p_mode == 0:  # for ProtoPNet
            f_x = self.conv_features(x)
            distances = self.prototype_distances(f_x)
            distances = distances.flatten(2)
            min_distances = distances.topk(self.topk_p, dim=2, largest=False)[0].mean(2)
        elif self.p_mode >= 1:  # for XProtoNet, MProtoNet and MAProtoNet
            f_x, p_map, h = self.conv_features(x)
            distances = self.prototype_distances(f_x, p_map)
            distances = distances.flatten(2)
            min_distances = distances.flatten(1)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        if self.p_mode >= 1 and self.training:
            return logits, min_distances, h, p_map
        elif self.training:
            return logits, min_distances
        else:
            return logits

    def push_forward(self, x):
        if self.p_mode == 0:  # for ProtoPNet
            f_x = self.conv_features(x)
            distances = self.prototype_distances(f_x)
            return f_x, distances
        elif self.p_mode >= 1:  # for XProtoNet, MProtoNet and MAProtoNet
            f_x, p_map, h = self.conv_features(x)
            distances = self.prototype_distances(f_x, p_map)
            return f_x, distances, p_map


    def __repr__(self):
        return (
            f"ProtoNets(\n"
            f"\tpmode: {self.p_mode},\n"
            f"\tfeatures: {self.backbone_name},\n"
            f"\timg_size: {self.in_size[1:]},\n"
            f"\tprototype_shape: {self.prototype_shape},\n"
            f"\tproto_layer_rf_info: {self.proto_layer_rf_info},\n"
            f"\tnum_classes: {self.num_classes},\n"
            f"\tepsilon: {self.epsilon},\n"
            f"\ttopk_p: {self.topk_p}\n"
            f")"
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = self.prototype_class_identity.mT
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        init_resnet3d(self.backbone, in_channels=self.in_size[0])
        init_conv(self.features_add_ons)
        if self.p_mode >= 1:  # for XProtoNet, MProtoNet and MAProtoNet
            init_conv(self.p_map)
        if self.p_mode >= 3:  # for MAProtoNet
            init_conv(self.scale_layers)
        self.set_last_layer_incorrect_connection(-0.5)
