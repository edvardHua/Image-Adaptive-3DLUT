# -*- coding: utf-8 -*-
# @Time : 2022/10/18 14:55
# @Author : zihua.zeng
# @File : model_edz.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ailut import ailut_transform


class BasicBlock(nn.Sequential):
    r"""Conv+LeakyReLU[+InstanceNorm].
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class TPAMIBackbone(nn.Sequential):
    r"""
    Args:
        pretrained (bool, optional): [ignored].
        input_resolution (int, optional): 输入分辨率
        extra_pooling (bool, optional): 最后加入 pooling 降低参数量
        weight_mode (bool, optional): 是否回归权重
        n_weights (int, optional): 权重个数
    """

    def __init__(self, pretrained=False, input_resolution=256, weight_mode=False, n_weights=3, extra_pooling=False):
        body = [
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            BasicBlock(32, 64, stride=2, norm=True),
            BasicBlock(64, 128, stride=2, norm=True),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]

        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))

        self.weight_mode = weight_mode
        if weight_mode:
            if extra_pooling:
                body.append(nn.Conv2d(128, n_weights, 2, padding=0))
            else:
                body.append(nn.Conv2d(128, n_weights, 8, padding=0))

        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
                             mode='bilinear', align_corners=False)
        tmp = super().forward(imgs)
        return tmp.view(imgs.shape[0], -1)


class LUTGenerator(nn.Module):
    r"""

    Args:
        n_colors (int): 输入通道数
        n_vertices (int): 3D LUT 顶点数，33，36，64 etc.
        n_feats (int): TPAMIBackbone 出来的通道数
        n_ranks (int): 多少个基础的 lut 数量， 3
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1 3, 3 * (33 * 3)
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
        ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class AiLUT(nn.Module):

    def __init__(self,
                 n_ranks=3,
                 n_vertices=33,
                 n_colors=3):
        super(AiLUT, self).__init__()
        self.backbone = TPAMIBackbone()
        self.lut_generator = LUTGenerator(
            n_colors, n_vertices,
            self.backbone.out_channels, n_ranks)

        uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1).repeat(n_colors, 1)
        self.register_buffer('uniform_vertices', uniform_vertices.unsqueeze(0))
        self.init_weights()

    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(special_initilization)
        self.lut_generator.init_weights()

    def forward(self, x):
        codes = self.backbone(x)
        weights, luts = self.lut_generator(codes)
        vertices = self.uniform_vertices
        outs = ailut_transform(x, luts, vertices)
        return outs, weights, luts, vertices


if __name__ == '__main__':
    dummy_inp = torch.randn((1, 3, 256, 256))
    ailut = AiLUT()
    out = ailut(dummy_inp)
    print(out[0].shape, out[1].shape, out[2].shape)

    # tpa = TPAMIBackbone(extra_pooling=True, weight_mode=True)
    # out = tpa(dummy_inp)
    # print(out.shape)
