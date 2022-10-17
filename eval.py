# -*- coding: utf-8 -*-
# @Time : 2022/8/8 2:56 PM
# @Author : zihua.zeng
# @File : test.py
import cv2
import numpy as np

from models import *

import torchvision.transforms.functional as TF


class Ada3DLUTV1:

    def __init__(self, model_path, lut_path):
        self.lut0 = Generator3DLUT_zero(dim=36)
        self.lut1 = Generator3DLUT_zero(dim=36)
        self.lut2 = Generator3DLUT_zero(dim=36)
        self.classifier = Classifier()
        self.trilinear_ = TrilinearInterpolation()

        LUTs = torch.load(lut_path, map_location="cpu")
        self.lut0.load_state_dict(LUTs["0"])
        self.lut1.load_state_dict(LUTs["1"])
        self.lut2.load_state_dict(LUTs["2"])
        self.lut0.eval()
        self.lut1.eval()
        self.lut2.eval()

        self.classifier.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.classifier.eval()

    def __call__(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = TF.to_tensor(frame)
        img = img.unsqueeze(0)
        LUT = self.__gen_lut(img)
        result = self.trilinear_.apply(LUT, img)

        ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    def __gen_lut(self, ndarr):
        pred = self.classifier(ndarr).squeeze()
        LUT = pred[0] * self.lut0.LUT + pred[1] * self.lut1.LUT + pred[
            2] * self.lut2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
        return LUT


if __name__ == '__main__':
    image_path = "/Users/zihua.zeng/Workspace/Image-Adaptive-3DLUT/demo_images/sRGB/1.png"
    model_path = "/Users/zihua.zeng/Workspace/Image-Adaptive-3DLUT/saved_models/_sRGB/classifier_163.pth"
    lut_path = "/Users/zihua.zeng/Workspace/Image-Adaptive-3DLUT/saved_models/_sRGB/LUTs_163.pth"

    enhancer = Ada3DLUTV1(model_path, lut_path)
    image = cv2.imread(image_path)
    out = enhancer(image)

    vs_img = np.hstack([image, out])
    cv2.imwrite("output_vs.jpg", vs_img)
