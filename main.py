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
# color for
# color_dics = np.asarray([[3.17784463e-04, 1.00986995e-07, 1.42690865e-07],
#                          [7.60518536e-02, 1.00000000e+00, 7.36885130e-01],
#                          [1.03331357e-01, 1.00000000e+00, 7.39237487e-01],
#                          [1.38145536e-01, 1.00000000e+00, 7.14086354e-01],
#                          [4.60065812e-01, 3.43642116e-01, 2.86048710e-01],
#                          [1.18432801e-07, 1.00000000e+00, 8.13184753e-02],
#                          [1.03293411e-01, 1.00000000e+00, 6.18309915e-01],
#                          [2.97720373e-01, 1.00000000e+00, 6.18937731e-01],
#                          [1.66859850e-01, 1.00000000e+00, 9.44254339e-01],
#                          [1.22655220e-01, 1.00000000e+00, 8.26458991e-01]])
# color_dics = np.clip((color_dics * 255), a_min=0, a_max=255).astype(np.uint8)
K = 50
iteration = 500
img_path = "test_imgs/cups.png"
# 360,260,3 requires 32GB mem!!!
img = cv2.imread(img_path) / 255.0  # H, W, C
# 587 440 3
#H, W, C = (120, 160, 3)
H, W, C = (352, 470, 3)
# H, W, C = img.shape
img = cv2.resize(img, (W, H))
print(int((H * W) ** 2 * 4 / 1024 / 1024 / 1024), "GB")
print("available mem:", int(psutil.virtual_memory().available/1024/1024/1024), "GB")
print("*" * 5, "extract colors", "*" * 5)
color_dics = extract_colors(img, K)  # K, N
X = img.reshape((-1, 3)).transpose((1, 0))  # N, 3 ->  3, N

print(X.shape, color_dics.shape)  # (3,N)   (3,K)
print("*" * 5, "SLRR", "*" * 5)
Phi_d, Wd, Ms = SLRR(X, color_dics, iteration=iteration)
Hlt_mask = Ms.transpose((1, 0))
Hlt_mask = Hlt_mask.reshape((H, W, 1))
Hlt_mask = np.clip((Hlt_mask * 255), a_min=0, a_max=255).astype(np.uint8)

cv2.imwrite("SLRR_results/Hlt.png", Hlt_mask)
diffuse = (np.dot(Phi_d, Wd)).transpose((1, 0))
diffuse = diffuse.reshape((H, W, 3))
diffuse = np.clip((diffuse * 255), a_min=0, a_max=255).astype(np.uint8)
cv2.imwrite("SLRR_results/diffuse.png", diffuse)
print("*" * 5, "Finish", "*" * 5)
