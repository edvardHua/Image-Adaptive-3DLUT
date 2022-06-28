#!/bin/bash
#
# 简单脚本，循环对文件夹下的图片做色彩增强
#

for i in {1..22}
do
python3 demo_eval.py --model_dir pretrained_models_0526 --image_name $i.png
done