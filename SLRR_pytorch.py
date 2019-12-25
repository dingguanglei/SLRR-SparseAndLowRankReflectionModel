# code=utf-8
# import numpy as np
import torch
from tqdm import tqdm
from functools import partial

gpu_id = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(gpu_id)
if device == "cuda": print(torch.cuda.get_device_name(0))
print("Use %s..." % gpu_id)
dtype = torch.float32
torch.set_default_dtype(dtype)

norm = torch.norm
inv = torch.inverse
sqrt = lambda x: x ** 0.5
sign = torch.sign
clip = torch.clamp

abs = torch.abs

ones = partial(torch.ones, device=device, dtype=dtype)
zeros = partial(torch.zeros, device=device, dtype=dtype)
eye = partial(torch.eye, device=device, dtype=dtype)


def svd(A):
    U, S, V = torch.svd(A)
    return U, S, V.T


def S_tau(x, tau):
    return sign(x) * clip(abs(x) - tau, min=0)


def max_0(x):
    return clip(x, min=0)


def dot(*arg):
    assert len(arg) >= 2
    res = arg[0]
    for x in arg[1:]:
        res = res.mm(x)
    return res


# =======================================
# init
def SLRR(X, color_dics, Gamma=None, iteration=200):
    """
    X (N, K)
    color_dics (3, K)
    """
    assert (X <= 1).all() and (color_dics <= 1).all()

    X = torch.from_numpy(X).float().to(device)  # 3, N
    Phi_d = torch.from_numpy(color_dics).float().to(device)  # 3, K

    N = X.shape[1]  # N,3
    K = color_dics.shape[1]  # 3 , K

    Wd = zeros((K, N))  # K, N
    J = zeros((K, N))  # K, N
    H = zeros((K, N))  # K, N
    Ms = zeros((K, N))  # 1, N
    S1 = zeros((K, N))
    S2 = zeros((1, N))
    Y1 = zeros((3, N))
    Y2 = zeros((K, N))
    Y3 = zeros((K, N))
    Y4 = zeros((K, N))
    Y5 = zeros((1, N))
    mu = 0.1
    mu_max = 1e10
    rho = 1.1  # p
    eps = 1e-6
    lamda = 0.1 / sqrt(N)
    tau = 1 / sqrt(N)

    is_converged = False
    if Gamma is None:
        Gamma = ones((3, 1)) * 1 / 3
        # Gamma[0] = 0.3
        # Gamma[1] = 0.5
        # Gamma[2] = 0.2
        Gamma = Gamma / norm(Gamma)
    # else:
    #     Gamma = ones((3, 1))
    #     Gamma[0] = 0.3
    #     Gamma[1] = 0.5
    #     Gamma[2] = 0.2
    # Converge Loop
    i = 0
    pbar = tqdm(total=iteration)
    #
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="log")
    # max_Es = []

    while not is_converged and i < iteration:
        i += 1
        pbar.update(1)
        # Update J
        J = update_J(N, Wd, Y2, eps, mu)
        # Update Ms
        Ms = update_Ms(Gamma, Phi_d, S2, Wd, X, Y1, Y5, lamda, mu)
        # Update H
        H = Update_H(Wd, Y3, mu, tau)
        # Update Wd
        Wd = Update_Wd(Gamma, H, J, K, Ms, Phi_d, S1, X, Y1, Y2, Y3, Y4, mu)
        # Update S1(K, N)S2(1, N)  Wd(K, N)  Y4(K, N)  Ms(1, N) Y5(1, N)
        S1 = max_0(Wd + Y4 / mu)
        S2 = max_0(Ms + Y5 / mu)
        # Update Ei, Yi
        E1 = X - dot(Phi_d, Wd) - dot(Gamma, Ms)  # E1(3, N)
        E2 = J - Wd  # E2(K, N)
        E3 = H - Wd  # E3(K, N)
        E4 = Wd - S1  # E4(K, N)
        E5 = Ms - S2  # E5(1, N)
        E = [E1, E2, E3, E4, E5]
        # print(E1.max(), E2.max(), E3.max(), E4.max(), E5.max())
        for Yi, Ei in zip([Y1, Y2, Y3, Y4, Y5], E):
            Yi += Ei * mu
        # print(Y1.max(), Y2.max(), Y3.max(), Y4.max(), Y5.max())
        # Update mu
        mu = min(mu_max, rho * mu)
        # is converged?
        X_norm = norm(X)
        max_E = max([norm(E[i]) / X_norm for i in range(len(E))])
        writer.add_scalar("max_E", max_E, i)
        is_converged = max_E < eps
    # print(max_E)
    writer.close()
    Phi_d = Phi_d.cpu().numpy()
    Wd = Wd.cpu().numpy()
    Ms = Ms.cpu().numpy()
    return Phi_d, Wd, Ms


def Update_Wd(Gama, H, J, K, Ms, Phi_d, S1, X, Y1, Y2, Y3, Y4, mu):
    # Wd(K, N) Phi_d(3, K) Gama(3, 1) Ms(1, N) ,J(K,N) H(K,N) S1(K,N) Y1(3 ,N) Y2(K, N) Y3(K, N) Y4(K,N)
    W1 = dot(Phi_d.T, Phi_d) + 3 * eye(K)  # W1 (3,3)
    W2 = dot(Phi_d.T, X) - dot(Phi_d.T, Gama, Ms) + J + H + S1 + (dot(Phi_d.T, Y1) + Y2 + Y3 - Y4) / mu  # (K, N)
    Wd = dot(inv(W1), W2)
    Wd = clip(Wd, min=0, max=3)
    return Wd


def Update_H(Wd, Y3, mu, tau):
    # (K, N)  Wd(K, N) Y3(K, N)
    H = S_tau(Wd - Y3 / mu, tau / mu)
    return H


def update_Ms(Gama, Phi_d, S2, Wd, X, Y1, Y5, lamda, mu):
    # (1, N)  X(3, N) Phi_d(3, K)  Wd(K, N)  Y1(3, N)  Y5(1, N) Ms(1, N) S2(1, N)
    g = dot(Gama.T, Gama)
    Ms1 = (dot(Gama.T, X - dot(Phi_d, Wd) + Y1 / mu) - Y5 / mu + S2) / g  # Ms(1, N)
    Ms = S_tau(Ms1, lamda / (mu * g))  # Ms(1, N)
    return Ms


def update_J(N, Wd, Y2, eps, mu):
    #    J(K, N)   Wd(K, N)   Y2(K, N)
    A = Wd - Y2 / mu  # A(K, N)
    U, sigma, VT = svd(A)  # U(K,K)  sigma(10,)   VT(N,N)
    w = sqrt(N) / (abs(sigma) + eps)  # W(10, 0)
    # sigma = sgn(sigma) * max_0(abs(sigma) - 1 / mu * w)  # sigma(10, )
    sigma = S_tau(sigma, 1 / mu * w)
    S = zeros((U.shape[1], VT.shape[0]))  # S(K, N)
    for i in range(len(sigma)):
        S[i][i] = sigma[i]
    J = dot(U, S, VT)  # (K, N)
    return J
