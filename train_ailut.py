# -*- coding: utf-8 -*-
# @Time : 2022/10/18 14:56
# @Author : zihua.zeng
# @File : train.py

# 1. 读取配置
# 2. 加载数据集
# 3. 加载模型和训练参数
# 4. 训练与评估

import sys
import json
import time
import datetime
import argparse
import itertools

from models import *
from datasets import *
from model_edz import *
from pathlib import Path
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=400, help="total number of epochs of training")
parser.add_argument("--dataset_path", type=str,
                    default="/Users/zihua.zeng/Dataset/色彩增强数据集/Apple_Enhance_sub/Apple_Enhance",
                    help="Training Dataset path")
parser.add_argument("--n_vertices", type=int, default=36)
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
parser.add_argument("--fixed_seed", type=str, default="False", help="whether fixed seed or not")

# 1. 读取配置，创建相关文件夹
opt = parser.parse_args()
if not opt.output_dir:
    opt.output_dir = time.strftime("%m-%d_%H_%M_%S") + "_ailut"
Path("saved_models/%s" % opt.output_dir).mkdir(parents=True, exist_ok=True)
Path("saved_models/%s/best_model" % opt.output_dir).mkdir(parents=True, exist_ok=True)
# 保存当前训练的配置
json.dump(vars(opt), open("saved_models/%s/best_model/config.json" % opt.output_dir, "w"))

# 2. 加载数据集
dataloader = DataLoader(
    ImageDataset_apple_sRGB(opt.dataset_path, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

psnr_dataloader = DataLoader(
    ImageDataset_apple_sRGB(opt.dataset_path, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# 3. 加载模型和训练参数

if eval(opt.fixed_seed):
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
criterion_pixelwise = torch.nn.MSELoss()

model = AiLUT(n_vertices=opt.n_vertices)

TV3 = TV_3D(dim=opt.n_vertices)
trilinear_ = TrilinearInterpolation()

if cuda:
    model = model.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

optimizer_G = torch.optim.Adam(
    itertools.chain(model.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2))


# 4. 训练与评估
def generator_train(img):
    out, weights, lut, _ = model(img)

    weights_norm = torch.mean(weights.squeeze() ** 2)
    return out.squeeze(), weights_norm, lut.squeeze()


def generator_eval(img):
    out, weights, lut, _ = model(img)
    weights_norm = torch.mean(weights.squeeze() ** 2)
    return out, weights_norm


def calculate_psnr():
    model.eval()
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


prev_time = time.time()
max_psnr = 0
max_epoch = 0
for epoch in range(opt.epoch, opt.n_epochs):
    mse_avg = 0
    psnr_avg = 0
    model.train()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        fake_B, weights_norm, lut = generator_train(real_A)

        # Pixel-wise loss
        mse = criterion_pixelwise(fake_B, real_B)

        tv_cons, mn_cons = TV3(lut.squeeze())

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
        torch.save(model.state_dict(), "saved_models/%s/best_model/ailut_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/best_model/result.txt' % opt.output_dir, 'w')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))
        file.flush()
        file.close()

    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(model.state_dict(), "saved_models/%s/ailut_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/result.txt' % opt.output_dir, 'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))
        file.flush()
        file.close()
