# -*- coding: utf-8 -*-
# @Time : 2022/7/18 6:28 PM
# @Author : zihua.zeng
# @File : infer.py

from __future__ import print_function, division

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# 30 个类别，这个顺序暂时不要改变，和模型输出的类别顺序绑定了
class_names = ['10_Waterfall', '11_Snow', '12_Landscape', '13_Underwater', '14_Architecture', '15_Sunset_Sunrise', '16_Blue_Sky', '17_Cloudy_Sky',
               '18_Greenery', '19_Autumn_leaves', '1_Portrait', '20_Flower', '21_Night_shot', '22_Stage_concert', '23_Fireworks', '24_Candle_light',
               '25_Neon_lights', '26_Indoor', '27_Backlight', '28_Text_Documents', '29_QR_images', '2_Group_portrait', '30_Computer_Screens', '3_Kids',
               '4_Dog', '5_Cat', '6_Macro', '7_Food', '8_Beach', '9_Mountain']


model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
# 场景分类有30个
model_ft.fc = nn.Linear(num_ftrs, 30)

ckpt = torch.load("CamSSD_Resnet18.pth", map_location="cpu")
model_ft.load_state_dict(ckpt['state_dict'])
model_ft.eval()

# 预处理
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


image = Image.open("2.jpg")
test = image_transform(image)
results = model_ft(test.unsqueeze(0))
_, preds = torch.max(results, 1)
print(class_names[preds[0]])
