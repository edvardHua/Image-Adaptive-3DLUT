# -*- coding: utf-8 -*-
# @Time : 2022/5/18 11:21 AM
# @Author : zihua.zeng
# @File : comparison.py

import os
import cv2
import numpy as np


def image_vs_compare(ori_path, alo_1, alo_2, alo_3, out_path):
    os.makedirs(out_path, exist_ok=True)

    for fn in os.listdir(ori_path):
        if not fn.endswith(".png"):
            continue
        img_ori = cv2.imread(os.path.join(ori_path, fn), cv2.IMREAD_UNCHANGED)[:,:,:3]
        img_alo1 = cv2.imread(os.path.join(alo_1, fn), cv2.IMREAD_UNCHANGED)[:,:,:3]
        img_alo2 = cv2.imread(os.path.join(alo_2, fn), cv2.IMREAD_UNCHANGED)[:,:,:3]
        img_alo3 = cv2.imread(os.path.join(alo_3, fn.replace(".png", ".jpg")), cv2.IMREAD_UNCHANGED)
        if img_ori.shape[0] != img_alo1.shape[0]:
            factor = img_alo1.shape[0] / img_ori.shape[0]
            res_width = int(img_ori.shape[1] * factor)
            img_ori = cv2.resize(img_ori, (res_width, img_alo1.shape[0]))

        img_vs = np.hstack([img_ori, img_alo1, img_alo2, img_alo3])
        cv2.imwrite(os.path.join(out_path, fn.replace(".png", ".jpg")), img_vs)


def single_video_to_img(input_path):
    suffix = ".MOV"
    out_suffix = ".png"
    for fn in os.listdir(input_path):
        if not fn.endswith(suffix):
            continue
        cap = cv2.VideoCapture(os.path.join(input_path, fn))
        _, frame = cap.read()
        h, w, _ = frame.shape
        cv2.imwrite(os.path.join(input_path, fn.replace(suffix, out_suffix)), frame)
        cap.release()


if __name__ == '__main__':
    image_vs_compare("/Users/zihua.zeng/Downloads/Test",
                     "demo_results",
                     "/Users/zihua.zeng/Workspace/PPR10K/code_3DLUT/results/mask_glc_b_-1",
                     "/Users/zihua.zeng/Desktop/topaz",
                     "vs_results")

    # single_img_to_video("/Users/zihua.zeng/Desktop")
    # single_video_to_img("/Users/zihua.zeng/Downloads/Test")
    pass
