# code=utf-8
"""
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24
# @Author  : Guanglei Ding
# @Site    : http://openaccess.thecvf.com/content_ECCV_2018/papers/Jie_Guo_Single_Image_Highlight_ECCV_2018_paper.pdf
# @File    : main.py
# @description: This is a NON-OFFICIAL python implementation of 'Single Image Highlight Removal with a Sparse and Low-Rank Reflection Model'
"""
import numpy as np
import cv2
from SLRR_pytorch import SLRR
from extract_colors import extract_colors
import psutil
import os

K = 50
iteration = 250
Gamma = np.ones((3, 1), dtype=np.float32) * 1 / 3  # (1 ,3)
img_read_path = "test_imgs/fig2_e1.png"
_, fullfilename = os.path.split(img_read_path)
filename, _ = os.path.splitext(fullfilename)
img_save_dir = os.path.join("SLRR_results", filename)
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)

img = cv2.imread(img_read_path)  # H, W, C
#H, W, C = (128, 128, 3)
H, W, C = img.shape  # 360,260,3 requires 32GB mem!!!
img = cv2.resize(img, (W, H))

print("Require mem :", int((H * W) ** 2 * 4 / 1024 / 1024 / 1024), "GB")
print("Available mem:", int(psutil.virtual_memory().available / 1024 / 1024 / 1024), "GB")
print("*" * 5, "extract colors", "*" * 5)

color_dics = extract_colors(img, K)  # (3,K)
X = img.astype(np.float).reshape((-1, 3)).T / 255.0  # N, 3 ->  3, N
print(X.shape, color_dics.shape)  # (3,N) ,  (3,K)

print("*" * 5, "SLRR", "*" * 5)
Phi_d, Wd, Ms = SLRR(X, color_dics, Gamma=Gamma, iteration=iteration)
Hlt = np.dot(Gamma, Ms).T
Hlt = Hlt.reshape((H, W, 3))
Hlt = np.clip(Hlt * 255, a_min=0, a_max=255).astype(np.uint8)
cv2.imwrite(os.path.join(img_save_dir, "Hlt.png"), Hlt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0.3])
diffuse = cv2.subtract(img, Hlt)
cv2.imwrite(os.path.join(img_save_dir, "diffuse.png"), diffuse, [int(cv2.IMWRITE_PNG_COMPRESSION), 0.3])
print("*" * 5, "Finish", "*" * 5)
