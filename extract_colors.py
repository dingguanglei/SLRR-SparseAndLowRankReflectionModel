# code=utf-8
# Final funciton extract colors
import numpy as np
from scipy.cluster.vq import kmeans

def extract_colors(original_img, K=50, iter=200):
    """extract the color dictionary by using Kmeans
    Params:
        original_img: input image(H,W,C)
        K: size of color dictionary. default:50
        iter: Kmeans iteration
    Return:
        colors: shape(3, K) range(0, 1)

    """
    # BGR
    img = original_img.copy().astype(np.float64).reshape(-1, 3)  # N, 3
    h, w, c = original_img.shape
    B = R = 0
    G = T = 1
    R = P = 2
    eps = 1e-8
    # convert to spherical coordinates
    img_rtp = np.empty_like(img)  # N, 3
    x = img[:, B]
    y = img[:, G]
    z = img[:, R]
    img_rtp[:, R] = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    img_rtp[:, T] = np.arccos(z / (img_rtp[:, R] + eps))
    img_rtp[:, P] = np.arctan(y / (x + eps))
    # cluster on T, P
    # K = 10
    cluster_tp, dist = kmeans(img_rtp[:, (T, P)], K, iter=iter, thresh=1e-05, check_finite=True)
    # min_index, cluster_tp = kmeans_numpy(img_rtp[:, (T, P)], K, iter=iter)

    # get R of cluster_tp
    img_tp = img_rtp[:, (T, P)][:, np.newaxis, :]  # N, 1, 2
    img_r = img_rtp[:, R]  # N,
    # print(x, img_rtp)
    distance = np.linalg.norm(cluster_tp - img_tp, axis=2)  # , N,K
    min_index = np.argmin(distance, axis=1).astype(np.uint32)  # ,N
    K = min(K, cluster_tp.shape[0])
    print(K)
    cluster_r = np.ones((K)).astype(np.float64)
    for i in range(K):
        # cluster_r[i] = (img_r[min_index == i].max() + img_r[min_index == i].min()) * 0.5
        cluster_r[i] = (img_r[min_index == i].min())
    cluster_r = cluster_r[..., np.newaxis]
    cluster_rtp = np.concatenate([cluster_r, cluster_tp], axis=1)
    # print(cluster_rtp)
    # convert to RGB
    colors = np.empty_like(cluster_rtp)
    colors[:, B] = cluster_rtp[:, R] * np.sin(cluster_rtp[:, T]) * np.cos(cluster_rtp[:, P])
    colors[:, G] = cluster_rtp[:, R] * np.sin(cluster_rtp[:, T]) * np.sin(cluster_rtp[:, P])
    colors[:, R] = cluster_rtp[:, R] * np.cos(cluster_rtp[:, T])
    colors = np.clip(colors, a_min=0, a_max=1)
    return colors.transpose([1, 0])


if __name__ == '__main__':
    import cv2

    img = cv2.imread("test_imgs/green.png")
    img = cv2.resize(img, (128, 128)) / 255.0
    colors = extract_colors(img, 50)
    print(colors.max(), colors.min())
    print(colors)
