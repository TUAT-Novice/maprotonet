# Implementation of resNet3d with quadruplet attention

# modify from source code of triplet attention:
# https://github.com/landskape-ai/triplet-attention/blob/master/MODELS/resnet.py

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.quadruplet_attention import QuadrupletAttention


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, use_quadruplet_attention=False
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_quadruplet_attention:
            self.quadruplet_attention = QuadrupletAttention()
        else:
            self.quadruplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.quadruplet_attention is None:
            out = self.quadruplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, use_quadruplet_attention=False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_quadruplet_attention:
            self.quadruplet_attention = QuadrupletAttention()
        else:
            self.quadruplet_attention = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.quadruplet_attention is not None:
            out = self.quadruplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv3d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool3d(7)
        else:
            self.conv1 = nn.Conv3d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, att_type=att_type
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, att_type=att_type
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, att_type=att_type
        )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode="fan_out")
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                use_quadruplet_attention=att_type == "QuadrupletAttention",
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    use_quadruplet_attention=att_type == "QuadrupletAttention",
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool3d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type):
    assert network_type in [
        "ImageNet",
        "CIFAR10",
        "CIFAR100",
    ], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101, 152], "network depth should be 18, 34, 50 or 101"

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    elif depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], network_type, num_classes, att_type)

    return model

