import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
# from trilinear_c._ext import trilinear
import trilinear


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


class Resnet18(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(Resnet18, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("resources/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("resources/IdentityLUT64.txt", 'r')
        elif dim == 36:
            file = open("resources/IdentityLUT36.txt", "r")

        LUT = file.readlines()
        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)

        for i in range(0, dim):
            # j 是高
            for j in range(0, dim):
                # k 是宽
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = LUT[n].split()
                    self.LUT[0, i, j, k] = float(x[0])
                    self.LUT[1, i, j, k] = float(x[1])
                    self.LUT[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        return self.TrilinearInterpolation.apply(self.LUT, x)


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        return self.TrilinearInterpolation.apply(self.LUT, x)


class TrilinearInterpolation(torch.autograd.Function):

    def forward(self, LUT, x):

        x = x.contiguous()
        output = x.new(x.size())
        dim = LUT.size()[-1]
        shift = dim ** 3
        binsize = 1.0001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        self.x = x
        self.LUT = LUT
        self.dim = dim
        self.shift = shift
        self.binsize = binsize
        self.W = W
        self.H = H
        self.batch = batch

        if x.is_cuda:
            if batch == 1:
                trilinear.forward(LUT, x, output, dim, shift, binsize, W, H, batch)
            elif batch > 1:
                output = output.permute(1, 0, 2, 3).contiguous()
                trilinear.forward(LUT, x.permute(1, 0, 2, 3).contiguous(), output, dim, shift, binsize,
                                  W, H, batch)
                output = output.permute(1, 0, 2, 3).contiguous()

        else:
            # trilinear.trilinear_forward(LUT, x, output, dim, shift, binsize, W, H, batch)
            trilinear.forward(LUT, x, output, dim, shift, binsize, W, H, batch)

        return output

    def backward(self, grad_x):

        grad_LUT = torch.zeros(3, self.dim, self.dim, self.dim, dtype=torch.float)

        if grad_x.is_cuda:
            grad_LUT = grad_LUT.cuda()
            if self.batch == 1:
                trilinear.backward(self.x, grad_x, grad_LUT, self.dim, self.shift, self.binsize, self.W,
                                   self.H, self.batch)
            elif self.batch > 1:
                trilinear.backward(self.x.permute(1, 0, 2, 3).contiguous(),
                                   grad_x.permute(1, 0, 2, 3).contiguous(), grad_LUT, self.dim,
                                   self.shift, self.binsize, self.W, self.H, self.batch)
        else:
            trilinear.backward(self.x, grad_x, grad_LUT, self.dim, self.shift, self.binsize, self.W, self.H,
                               self.batch)

        return grad_LUT, None


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


if __name__ == '__main__':
    from torchprofile import profile_macs

    # cls = Classifier()
    # inp = torch.rand((1, 3, 256, 256))
    # out = cls(inp)
    # print(inp.shape, out.shape)
    # macs = profile_macs(cls, inp)
    # # 0.074 GFLOPs
    # print(macs / 1e9)

    # 测试一下 TV_3D 这个模块
    # tv = TV_3D()
    # dummy_inp = Generator3DLUT_zero()
    # print(dummy_inp.LUT.shape)
    #
    # out = tv(dummy_inp)
    # print(out[0].shape, out[1].shape)

    # 测试 resent 网络
    dummy_inp = torch.randn((1, 3, 224, 224))
    resnet = Resnet18(out_dim=3)
    out = resnet(dummy_inp)
    macs = profile_macs(resnet, dummy_inp)
    print(macs / 1e9)

    # classifier = Classifier()
    # classifier.load_state_dict(torch.load("resources/pretrained_models/sRGB/classifier.pth", map_location="cpu"))
    # classifier.eval()
    # torch.onnx.export(classifier, dummy_inp, "weight_predictor.onnx", )
