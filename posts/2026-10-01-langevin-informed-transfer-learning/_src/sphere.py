"""Cobalt nanoparticle on S^2: reference spectrum (Witten-Laplacian in real SH) and LITL (fixed polynomial
dictionary, step 2 of LITL) from uniform samples + black-box energy evaluations."""
import numpy as np
from scipy.special import sph_harm_y
from scipy.linalg import eigh
from itertools import product

def U(m, Ku=1.0, Kc=0.3):
    x, y, z = m[..., 0], m[..., 1], m[..., 2]
    return -Ku * z**2 + Kc * (x*x*y*y + y*y*z*z + z*z*x*x)

def gradU(m, Ku=1.0, Kc=0.3):
    x, y, z = m[..., 0], m[..., 1], m[..., 2]
    g = np.stack([2*Kc*x*(y*y+z*z), 2*Kc*y*(x*x+z*z), -2*Ku*z + 2*Kc*z*(x*x+y*y)], -1)
    return g

def sgradU(m, Ku=1.0, Kc=0.3):
    g = gradU(m, Ku, Kc)
    return g - (g*m).sum(-1, keepdims=True)*m

def lapS_U(m, Ku=1.0, Kc=0.3):
    x, y, z = m[..., 0], m[..., 1], m[..., 2]
    q = x*x*y*y + y*y*z*z + z*z*x*x
    return Ku*(-2 + 6*z*z) + Kc*(4 - 20*q)

# ---------------- reference: H = -beta^{-1} Lap_S + (beta/4)|grad_S U|^2 - (1/2) Lap_S U, real SH basis
def real_sh(lmax, theta, phi):
    cols, ls = [], []
    for l in range(lmax+1):
        for mm in range(-l, l+1):
            Y = sph_harm_y(l, abs(mm), theta, phi)
            if mm > 0: v = np.sqrt(2)*(-1)**mm*Y.real
            elif mm < 0: v = np.sqrt(2)*(-1)**mm*Y.imag
            else: v = Y.real
            cols.append(v); ls.append(l)
    return np.stack(cols, -1), np.array(ls)

def reference(beta, Ku=1.0, Kc=0.3, lmax=28, k=12):
    nt, npp = 2*lmax+20, 4*lmax+40
    xg, wg = np.polynomial.legendre.leggauss(nt)
    theta = np.arccos(xg); phi = np.arange(npp)*2*np.pi/npp
    T, P = np.meshgrid(theta, phi, indexing='ij')
    W = (wg[:, None]*np.ones(npp)[None, :])*(2*np.pi/npp)
    m = np.stack([np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)], -1)
    Y, ls = real_sh(lmax, T.ravel(), P.ravel())
    w = W.ravel(); mm = m.reshape(-1, 3)
    g = sgradU(mm, Ku, Kc)
    Veff = beta/4*(g*g).sum(-1) - 0.5*lapS_U(mm, Ku, Kc)
    H = (Y*(w*Veff)[:, None]).T @ Y + np.diag(ls*(ls+1)/beta)
    ev, V = eigh(H)
    return -ev[:k], V[:, :k], (Y, ls, mm, w)

# ---------------- LITL with fixed polynomial dictionary
def monomials(P):
    exps = [e for e in product(range(P+1), repeat=3) if sum(e) in (P, P-1)]
    return np.array(exps)

def features(m, exps):
    x, y, z = m[:, 0:1], m[:, 1:2], m[:, 2:3]
    return (x**exps[:, 0])*(y**exps[:, 1])*(z**exps[:, 2])

def feature_grads(m, exps):
    out = []
    for k in range(3):
        e = exps.copy(); c = e[:, k].astype(float); e[:, k] = np.maximum(e[:, k]-1, 0)
        out.append(features(m, e)*c)
    G = np.stack(out, 1)  # n x 3 x F
    return G - m[:, :, None]*(m[:, :, None]*G).sum(1, keepdims=True)  # tangential

def litl(Msamp, Uvals, beta, P=8, rank=None, wts=None):
    """Step 2 of LITL: Galerkin from (uniform) samples + black-box energies. Returns lam, coefficient matrix, ..."""
    exps = monomials(P)
    Z = features(Msamp, exps); J = feature_grads(Msamp, exps)
    v = np.exp(-beta*(Uvals - Uvals.min()))
    if wts is not None: v = v*wts
    v = v/v.sum()
    C = (Z*v[:, None]).T @ Z
    D = np.einsum('n,nki,nkj->ij', v, J, J)/beta
    # whiten C (may be ill-conditioned)
    s, Q = eigh(C)
    keep = s > s.max()*1e-12
    T = Q[:, keep]/np.sqrt(s[keep])
    kap, Vw = eigh(T.T @ D @ T)
    coef = T @ Vw            # C-orthonormal eigenfunctions
    lam = -kap
    # drift weights: G(m) = -(I-mm^T) sum_i lam_i psi_i(m) E_pi[psi_i m]
    Em = (Z @ coef * v[:, None]).T @ Msamp     # F' x 3
    return dict(lam=lam, coef=coef, exps=exps, Em=Em)

def litl_eval(model, m, idx):
    return features(m, model['exps']) @ model['coef'][:, idx]

def litl_drift(model, m, r):
    Z = features(m, model['exps']) @ model['coef'][:, :r+1]
    G = -(Z*model['lam'][:r+1]) @ model['Em'][:r+1]
    return G - (G*m).sum(-1, keepdims=True)*m

def unif(n, rng):
    x = rng.standard_normal((n, 3)); return x/np.linalg.norm(x, axis=1, keepdims=True)

def fib(n):
    i = np.arange(n)+0.5; z = 1-2*i/n; r = np.sqrt(1-z*z); ph = np.pi*(1+5**0.5)*i
    return np.stack([r*np.cos(ph), r*np.sin(ph), z], -1)

if __name__ == "__main__":
    lam, _, _ = reference(1.0)
    print("reference beta=1:", np.round(lam, 4))
    rng = np.random.default_rng(0)
    for n in [1000, 10000]:
        M = unif(n, rng)
        mod = litl(M, U(M), 1.0, P=8)
        print("LITL n=%d P=8:" % n, np.round(mod['lam'][:10], 4))
    Mf = fib(20000)
    for P in [6, 8, 10]:
        mod = litl(Mf, U(Mf), 1.0, P=P)
        print("quadrature P=%d" % P, np.round(mod['lam'][:10], 4))
    for beta in [2, 4, 6, 8]:
        lam, _, _ = reference(beta)
        out = [np.round(lam[:6], 4)]
        for P in [8, 10, 12]:
            mod = litl(Mf, U(Mf), beta, P=P); out.append(np.round(mod['lam'][:4], 4))
        print("beta", beta, *out)
