"""Minimal SGOT re-implementation (linear kernel, primal RRR) used for the blog figures."""
import numpy as np
from scipy.linalg import eigh, eig
from scipy.optimize import linear_sum_assignment


def hankel(s, w):
    n = len(s) - w
    idx = np.arange(w)[None, :] + np.arange(n)[:, None]
    return s[idx], s[idx + 1]


def rrr(s, fs, rank, window_sec=1.0, reg=1e-8):
    w = int(round(window_sec * fs))
    X, Y = hankel(s, w)
    n = X.shape[0]
    C = X.T @ X / n + reg * np.eye(w)
    Cxy = X.T @ Y / n
    sig2, V = eigh(Cxy @ Cxy.T, C)
    V = V[:, ::-1][:, :rank]                      # V^T C V = I
    G = V @ (V.T @ Cxy)                            # w x w RRR estimator
    M = V.T @ Cxy @ V
    mu, wl, wr = eig(M, left=True, right=True)
    psi = V @ wr                                   # right eigvecs of G
    xi = Cxy.T @ (V @ wl)                          # left eigvecs of G (xi^H G = mu xi^H)
    lam = np.log(mu.astype(complex)) * fs          # generator eigenvalues (1/s)
    return dict(G=G, mu=mu, lam=lam, psi=psi, xi=xi, fs=fs)


def atoms(op):
    lam = op["lam"]
    tau, freq = lam.real, lam.imag / (2 * np.pi)
    psi = op["psi"] / np.linalg.norm(op["psi"], axis=0)
    xi = op["xi"] / np.linalg.norm(op["xi"], axis=0)
    return tau, freq, psi, xi


def cost_matrix(op1, op2, eta):
    t1, f1, p1, x1 = atoms(op1)
    t2, f2, p2, x2 = atoms(op2)
    dval = np.sqrt((t1[:, None] - t2[None, :]) ** 2 + (f1[:, None] - f2[None, :]) ** 2)
    # <psi xi^H, psi' xi'^H>_HS = (psi^H psi')(xi'^H xi)
    ip = (p1.conj().T @ p2) * (x2.conj().T @ x1).T
    dG = np.sqrt(np.clip(2 - 2 * np.abs(ip) ** 2, 0, None))
    return eta * dval + (1 - eta) * dG, dval, dG


def w1(C):
    r, c = linear_sum_assignment(C)
    return C[r, c].mean()


def sgot(op1, op2, eta=0.5):
    return w1(cost_matrix(op1, op2, eta)[0])


def all_metrics(op1, op2, eta=0.5):
    C, dval, dG = cost_matrix(op1, op2, eta)
    D = op1["G"] - op2["G"]
    return dict(hs=np.linalg.norm(D, "fro"), op=np.linalg.norm(D, 2),
                eig=w1(dval), sub=w1(dG), sgot=w1(C))


def oscillators(freqs, decays, fs, T=20.0, noise=1e-2, seed=0, amps=None):
    t = np.arange(int(T * fs) + 1) / fs
    amps = np.ones(len(freqs)) if amps is None else amps
    s = sum(a * np.exp(-d * t) * np.sin(2 * np.pi * f * t) for f, d, a in zip(freqs, decays, amps))
    rng = np.random.default_rng(seed)
    return t, s + noise * rng.standard_normal(len(t))
