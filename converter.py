# -*- coding: utf-8 -*-
# @Time : 2022/5/26 3:00 PM
# @Author : zihua.zeng
# @File : convert_model.py
#
# 转换模型和序列化lut值到txt
#

from models import *
import coremltools as ct


def model_to_coreml():
    """
    用来把模型转换成 coreml，方便快速测试在手机上的效果
    """
    classifier = Classifier()
    classifier.load_state_dict(torch.load("resources/classifier.pth", map_location="cpu"))
    classifier.eval()

    example_input = torch.rand(1, 3, 256, 256)
    traced_model = torch.jit.trace(classifier, example_input)

    image_input = ct.ImageType(shape=(1, 3, 256, 256),
                               channel_first=True,
                               scale=1 / 255.)
    model = ct.convert(
        traced_model,
        inputs=[image_input]
    )
    model.save("classifier.mlmodel")


def lut_tesnor_to_text():
    """
    把学习到的 3 个 3dlut 序列化成字符串保存
    """
    LUTs = torch.load("%s/LUTs.pth" % "pretrained_models/sRGB", map_location="cpu")
    LUT0 = Generator3DLUT_identity()
    LUT1 = Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()

    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])

    LUT0 = (LUT0.eval()).LUT
    LUT1 = (LUT1.eval()).LUT
    LUT2 = (LUT2.eval()).LUT

    c, c1, h, w = LUT0.shape

    # 每一行代表的是 33x33x33 的行号
    # 而列则代表的是 3，分别是 r, g, b

    for i, lut in enumerate([LUT0, LUT1, LUT2]):
        outf_union = open("resources/lut%d.txt" % i, "w")
        # red, green, blue
        for ci in range(c):
            for c1i in range(c1):
                for hi in range(h):
                    for wi in range(w):
                        outf_union.write("%.6f " % (lut[ci, c1i, hi, wi]))

        outf_union.close()


if __name__ == '__main__':
    model_to_coreml()
    # lut_tesnor_to_text()