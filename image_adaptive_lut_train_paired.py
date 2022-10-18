import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=400, help="total number of epochs of training")
parser.add_argument("--dataset_path", type=str,
                    default="/Users/zihua.zeng/Dataset/色彩增强数据集/Apple_Enhance_sub/Apple_Enhance",
                    help="Training Dataset path")
parser.add_argument("--model_type", type=str, default="lite", help="lite, resnet")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--output_dir", type=str, default="",
                    help="path to save model")
parser.add_argument("--test_mode", type=str, default="False", help="whether test or not")

opt = parser.parse_args()

if not opt.output_dir:
    opt.output_dir = time.strftime("%m-%d_%H_%M_%S")

if not eval(opt.test_mode):
    opt.output_dir = opt.output_dir + '_tpami'
    Path("saved_models/%s" % opt.output_dir).mkdir(parents=True, exist_ok=True)
    Path("saved_models/%s/best_model" % opt.output_dir).mkdir(parents=True, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity(dim=36)
LUT1 = Generator3DLUT_zero(dim=36)
LUT2 = Generator3DLUT_zero(dim=36)
# LUT3 = Generator3DLUT_zero()
# LUT4 = Generator3DLUT_zero()

classifier = None
if opt.model_type == "lite":
    classifier = Classifier()
elif opt.model_type == "resnet":
    classifier = Resnet18()

TV3 = TV_3D(dim=36)
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    # LUT3 = LUT3.cuda()
    # LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    # Load pretrained models
    LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # LUT3.load_state_dict(LUTs["3"])
    # LUT4.load_state_dict(LUTs["4"])
    classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))
else:
    # Initialize weights
    classifier.apply(weights_init_normal_classifier)
    torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

# Optimizers

optimizer_G = torch.optim.Adam(
    itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), lr=opt.lr,
    betas=(opt.b1, opt.b2))  # , LUT3.parameters(), LUT4.parameters()

dataloader = DataLoader(
    # ImageDataset_PPR10K_sRGB("ppr_10k", mode="train"),
    ImageDataset_apple_sRGB(opt.dataset_path, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

psnr_dataloader = DataLoader(
    # ImageDataset_PPR10K_sRGB("ppr_10k", mode="test"),
    ImageDataset_apple_sRGB(opt.dataset_path, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


def generator_train(img):
    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A0 = LUT0(img)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    # gen_A3 = LUT3(img)
    # gen_A4 = LUT4(img)

    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b, :, :, :] = pred[b, 0] * gen_A0[b, :, :, :] + pred[b, 1] * gen_A1[b, :, :, :] + pred[b, 2] * gen_A2[
                                                                                                                 b, :,
                                                                                                                 :, :]

    return combine_A, weights_norm


def generator_eval(img):
    pred = classifier(img).squeeze()

    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    weights_norm = torch.mean(pred ** 2)

    combine_A = trilinear_.apply(LUT, img)

    return combine_A, weights_norm


def calculate_psnr():
    classifier.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, weights_norm = generator_eval(real_A)
        fake_B = torch.round(fake_B * 255)
        real_B = torch.round(real_B * 255)
        mse = criterion_pixelwise(fake_B, real_B)
        # sometime mse will equal to zero
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr

    return avg_psnr / len(psnr_dataloader)


def visualize_result(epoch):
    """Saves a generated sample from the validation set"""
    classifier.eval()
    os.makedirs("images/%s/" % opt.output_dir + str(epoch), exist_ok=True)
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, weights_norm = generator_eval(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        fake_B = torch.round(fake_B * 255)
        real_B = torch.round(real_B * 255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        save_image(img_sample, "images/%s/%s/%s.jpg" % (opt.output_dir, epoch, img_name[0] + '_' + str(psnr)[:5]),
                   nrow=3, normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
max_psnr = 0
max_epoch = 0
for epoch in range(opt.epoch, opt.n_epochs):
    mse_avg = 0
    psnr_avg = 0
    classifier.train()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        fake_B, weights_norm = generator_train(real_A)

        # Pixel-wise loss
        mse = criterion_pixelwise(fake_B, real_B)

        tv0, mn0 = TV3(LUT0.LUT)
        tv1, mn1 = TV3(LUT1.LUT)
        tv2, mn2 = TV3(LUT2.LUT)
        # tv3, mn3 = TV3(LUT3)
        # tv4, mn4 = TV3(LUT4)
        tv_cons = tv0 + tv1 + tv2  # + tv3 + tv4
        mn_cons = mn0 + mn1 + mn2  # + mn3 + mn4

        loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons

        psnr_avg += 10 * math.log10(1 / mse.item())

        mse_avg += mse.item()

        loss.backward()

        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f, tv: %f, wnorm: %f, mn: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), psnr_avg / (i + 1), tv_cons, weights_norm, mn_cons, time_left,
               )
        )

    avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
        # 当前 psnr 值最高的单独放一个文件夹
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(),
                "2": LUT2.state_dict()}  # ,"3": LUT3.state_dict(),"4": LUT4.state_dict()
        torch.save(LUTs, "saved_models/%s/best_model/LUTs_%d.pth" % (opt.output_dir, epoch))
        torch.save(classifier.state_dict(), "saved_models/%s/best_model/classifier_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/best_model/result.txt' % opt.output_dir, 'w')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))
        file.flush()
        file.close()

    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(),
                "2": LUT2.state_dict()}  # ,"3": LUT3.state_dict(),"4": LUT4.state_dict()
        torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/result.txt' % opt.output_dir, 'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))
        file.flush()
        file.close()
