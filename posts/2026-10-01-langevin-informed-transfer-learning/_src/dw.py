"""1D double well: LITL step 2 with a fixed Gaussian-RBF dictionary on biased Langevin samples."""
import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal

beta = 1.0
U  = lambda x: 4*(x**2-1)**2
dU = lambda x: 16*x*(x**2-1)
A, s = 4.5, 0.35
V  = lambda x: -A*np.exp(-x**2/(2*s*s))
dV = lambda x: A*x/(s*s)*np.exp(-x**2/(2*s*s))

def reference(Upot, k=6, a=-2.2, b=2.2, N=4000):
    x = np.linspace(a, b, N); h = x[1]-x[0]
    u = Upot(x); xm = 0.5*(x[1:]+x[:-1])
    w = np.exp(-beta*Upot(xm))                    # edge weights
    p = np.exp(-beta*u)
    # symmetric form: -L ~ D^{-1/2} K D^{-1/2},  K from Dirichlet form with weights
    diag = np.zeros(N); diag[:-1] += w; diag[1:] += w
    diag = diag/(beta*h*h); off = -w/(beta*h*h)
    d = diag/p; o = off/np.sqrt(p[:-1]*p[1:])
    ev, vec = eigh_tridiagonal(d, o, select='i', select_range=(0, k-1))
    psi = vec/np.sqrt(p)[:, None]
    psi /= np.sqrt((psi**2*p[:, None]).sum(0)/p.sum())
    return x, -ev, psi

def simulate(dpot, n, dt=1e-3, thin=50, seed=0):
    rng = np.random.default_rng(seed); x = np.zeros(200); out = []
    while len(out)*200 < n:
        for _ in range(thin):
            x = x - dpot(x)*dt + np.sqrt(2*dt/beta)*rng.standard_normal(x.shape)
        out.append(x.copy())
    return np.concatenate(out)[:n]

centers = np.linspace(-1.5, 1.5, 16); width = 0.22
def feats(x):
    Z = np.exp(-(x[:, None]-centers)**2/(2*width**2))
    J = -(x[:, None]-centers)/width**2*Z
    return np.c_[np.ones_like(x), Z], np.c_[np.zeros_like(x), J]

def litl_1d(x, logw, reg=1e-8):
    v = np.exp(logw - logw.max()); v /= v.sum()
    Z, J = feats(x)
    C = (Z*v[:, None]).T @ Z; D = (J*v[:, None]).T @ J/beta
    C += reg*np.eye(len(C))*np.trace(C)/len(C)
    kap, cf = eigh(D, C)
    lam = -kap
    Ex = (Z @ cf*v[:, None]).T @ x
    return dict(lam=lam, cf=cf, Ex=Ex)

def psi_eval(mod, x, i): return feats(x)[0] @ mod['cf'][:, i]
def drift(mod, x, r):
    Z = feats(x)[0] @ mod['cf'][:, :r+1]
    return -(Z*mod['lam'][:r+1]) @ mod['Ex'][:r+1]     # = estimate of U'(x)

if __name__ == "__main__":
    xg, lt, pt = reference(U); _, lb, _ = reference(lambda x: U(x)+V(x))
    print("target ref", np.round(lt[:5], 3)); print("biased ref", np.round(lb[:5], 3))
    xs = simulate(lambda x: dU(x)+dV(x), 20000, seed=1)
    m_w = litl_1d(xs, beta*V(xs)); m_n = litl_1d(xs, 0*xs)
    print("LITL", np.round(m_w['lam'][:5], 3)); print("naive", np.round(m_n['lam'][:5], 3))
    xt = simulate(dU, 20000, seed=2)
    print("unbiased-traj frac in barrier |x|<0.4:", np.mean(np.abs(xt) < .4), " biased:", np.mean(np.abs(xs) < .4))
