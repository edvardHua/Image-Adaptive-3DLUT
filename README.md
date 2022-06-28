## Image-Adaptive-3DLUT

Ai 色彩增强代码训练仓库

***

## 仓库文件目录

```bash
# zihua.zeng @ C02G65LNMD6M in ~/Workspace/Image-Adaptive-3DLUT on git:master x [10:32:52] 
$ tree -L 1 .
.
├── demo_images # 测试图片
├── utils # 一些函数
├── resources # 资源文件夹，放着预训练模型等内容
├── trilinear_c # 3DLUT 三线性插值的 c 和 cpp 实现
├── trilinear_cpp # ---
├── image_adaptive_lut_evaluation.py # 评估指标
├── image_adaptive_lut_train_paired.py # 模型训练
├── models.py # 模型定义
├── datasets.py # 数据集
├── video_demo_eval.py # 测试在视频上的增强效果
├── torchvision_x_functional.py # pytorch 一些函数的封装
├── demo_eval.py # 测试在图片上的效果
├── converter.py # 转换模型，转换参数，用于在 ios 上运行
└── README.md 
```

***

## 参考链接

https://arxiv.org/abs/2009.14468
