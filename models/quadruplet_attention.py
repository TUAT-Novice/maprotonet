# Implementation of our quadruplet attention

# modify from: https://github.com/landskape-ai/triplet-attention/blob/master/MODELS/triplet_attention.py

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class QuadrupletAttention(nn.Module):
    def __init__(
        self,
        no_spatial=False,
    ):
        super(QuadrupletAttention, self).__init__()
        self.ChannelGateCWD = SpatialGate()
        self.ChannelGateCHD = SpatialGate()
        self.ChannelGateCHW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W, D)
        Returns:
        """
        # CWD branch
        x_perm1 = x.permute(0, 2, 1, 3, 4).contiguous()
        x_out1 = self.ChannelGateCWD(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3, 4).contiguous()
        # CHD branch
        x_perm2 = x.permute(0, 3, 2, 1, 4).contiguous()
        x_out2 = self.ChannelGateCHD(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1, 4).contiguous()
        # CHW branch
        x_perm3 = x.permute(0, 4, 2, 3, 1).contiguous()
        x_out3 = self.ChannelGateCHW(x_perm3)
        x_out31 = x_out3.permute(0, 4, 2, 3, 1).contiguous()
        # HWD branch
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 4) * (x_out + x_out11 + x_out21 + x_out31)
        else:
            x_out = (1 / 3) * (x_out11 + x_out21 + x_out31)
        return x_out
