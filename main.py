import numpy as np
import cv2
from update import SLRR

# color for
color_dics = np.asarray([[3.17784463e-04, 1.00986995e-07, 1.42690865e-07],
                         [7.60518536e-02, 1.00000000e+00, 7.36885130e-01],
                         [1.03331357e-01, 1.00000000e+00, 7.39237487e-01],
                         [1.38145536e-01, 1.00000000e+00, 7.14086354e-01],
                         [4.60065812e-01, 3.43642116e-01, 2.86048710e-01],
                         [1.18432801e-07, 1.00000000e+00, 8.13184753e-02],
                         [1.03293411e-01, 1.00000000e+00, 6.18309915e-01],
                         [2.97720373e-01, 1.00000000e+00, 6.18937731e-01],
                         [1.66859850e-01, 1.00000000e+00, 9.44254339e-01],
                         [1.22655220e-01, 1.00000000e+00, 8.26458991e-01]])
# color_dics = np.clip((color_dics * 255), a_min=0, a_max=255).astype(np.uint8)

X = cv2.imread("test_imgs/green.png")
X = cv2.resize(X, (64, 64)).reshape((-1, 3)).transpose((1, 0))  # 3, N
X = X / 255.0
color_dics = color_dics.transpose((1, 0))
print(X.shape, color_dics.shape)

Phi_d, Wd, Ms = SLRR(X, color_dics)
Hlt_mask = Ms.reshape((64, 64, 1))
Hlt_mask = np.clip((Hlt_mask * 255), a_min=0, a_max=255).astype(np.uint8)
cv2.imwrite("Hlt.png", Hlt_mask)

diffuse = (np.dot(Phi_d, Wd)).transpose((1, 0)).reshape((64, 64, 3))
diffuse = np.clip((diffuse * 255), a_min=0, a_max=255).astype(np.uint8)
cv2.imwrite("diffuse.png", diffuse)
