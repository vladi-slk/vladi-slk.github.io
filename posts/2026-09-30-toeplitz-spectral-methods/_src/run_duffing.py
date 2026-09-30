"""Re-run the Duffing experiments of the NLAA Toeplitz paper (from tklearn's notebook) and cache the results."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import os
sys.path.insert(0, os.environ.get("TKLEARN_SRC", str(Path(__file__).resolve().parent / "tklearn" / "src")))
import numpy as np
from kooplearn.datasets import DuffingOscillator
from tklearn.base import delay_embedding, predict, modes
from tklearn.estimators.primal_reduced_rank import fit_generator_filter
from tklearn.spectral_filters import (ExponentialFilter, SineHyperbolicFilter, GeneratorInverseFourier,
                                      TransferOperatorResolvent, GeneratorResolvent)
from tklearn.numerical_solvers import qr_solve_hpd

out = Path(__file__).parent

def feature_map(X, order=1):
    Z = X
    for i in range(1, order):
        Z = np.concatenate((Z, X ** (i + 1)), axis=1)
    return Z

poly_order = 5
t0 = time.time()
# ---------------- periodic regime ----------------
dt, burnin, ntr, nte = 0.1, 500, 8000, 1000
np.random.seed(14)
duffing = DuffingOscillator(alpha=0.5, beta=0.625, gamma=2, delta=1.5, omega=1, dt=dt)
sigma = 0.3
sol = duffing.sample(np.array([0.0, 0.0]), T=burnin + ntr + nte)[1 + burnin:]
ctx = 10
data = delay_embedding(sol, memory_length=ctx - 1)
n = ntr
flt = [ExponentialFilter(n, 1, dt), SineHyperbolicFilter(n, 1, dt),
       GeneratorInverseFourier(n, n - 1, dt, min_freq=0.1, max_freq=1)]
ranks = [5, 10, None]
trials = 10
test_traj = data[ntr:]
t_steps = test_traj.shape[0] // 2
preds = np.zeros((len(flt), len(ranks), trials, t_steps, 2))
evals = {}
noisy_example = None
for mi, model in enumerate(flt):
    for ri, rank in enumerate(ranks):
        for trial in range(trials):
            np.random.seed(trial)
            sol_noisy = sol + sigma * np.random.randn(*sol.shape)
            data_noisy = delay_embedding(sol_noisy, memory_length=ctx - 1)
            train = data_noisy[:ntr]
            if noisy_example is None:
                noisy_example = train[:, -2:].copy()
            Z = feature_map(train, order=poly_order)
            fmean = Z.mean(axis=0, keepdims=True)
            evd = fit_generator_filter(model, Z, dt=dt, tikhonov_reg=1e-12, rank=rank, centered=True)
            kmd = modes(eig_result=evd, initial_conditions=feature_map(test_traj[0][None, :], order=poly_order) - fmean,
                        obs_train=train[:, -2:])
            preds[mi, ri, trial] = predict(t=dt * np.arange(t_steps), mode_result=kmd).real
            if trial == 0 and rank is None:
                w = np.linalg.norm(kmd['modes'] * kmd['conditioning'], axis=1)[:-1]
                evals[mi] = (np.asarray(evd['values']), w)
        print(f"periodic model {mi} rank {rank} done  {time.time()-t0:.1f}s", flush=True)
np.savez(out / "periodic.npz", preds=preds, truth=test_traj[:t_steps, -2:], dt=dt, clean=data[:ntr, -2:],
         noisy=noisy_example, ev0=evals[0][0], w0=evals[0][1], ev1=evals[1][0], w1=evals[1][1],
         ev2=evals[2][0], w2=evals[2][1])

# ---------------- chaotic regime ----------------
np.random.seed(14)
duffing = DuffingOscillator(alpha=-1, beta=1, gamma=0.5, delta=0.3, omega=1, dt=dt)
sigma = 0.01
sol = duffing.sample(np.array([0.0, 0.0]), T=burnin + ntr + nte)[1 + burnin:]
sol_noisy = sol + sigma * np.random.randn(*sol.shape)
data = delay_embedding(sol, memory_length=ctx - 1)
data_noisy = delay_embedding(sol_noisy, memory_length=ctx - 1)
train = data_noisy[:ntr]
Zc = feature_map(train, order=poly_order)
fmean = Zc.mean(axis=0, keepdims=True)
n = train.shape[0]
obs = train[:, -1]

def resolvent_response(shift, kmd):
    k = kmd.copy()
    ev = np.log(1 / (shift + kmd["decay_rates"] - 1j * kmd["frequencies"] * 2 * np.pi))
    k["decay_rates"] = -np.real(ev)
    k["frequencies"] = np.imag(ev) / (2 * np.pi)
    return np.sqrt((np.abs(predict(1., k)) ** 2).mean())

shift = 0.01
a, npts = 5 / (2 * np.pi), 100
thetas = np.concatenate([np.linspace(-a, 0, npts // 2), np.linspace(0, a, npts // 2)])
evd = fit_generator_filter(ExponentialFilter(n, 1, dt), Zc, dt=dt, tikhonov_reg=0, rank=None, centered=True,
                           toeplitz_method="full")
kmd = modes(eig_result=evd, initial_conditions=Zc - fmean, obs_train=obs)
r_koop = np.array([resolvent_response(shift + 1j * th * 2 * np.pi, kmd) for th in thetas])
print(f"chaotic koopman done {time.time()-t0:.1f}s", flush=True)
r_to, r_gen = [], []
for th in thetas:
    f1 = TransferOperatorResolvent(n, n // 2, dt, shift=np.exp(shift + 1j * th * 2 * np.pi), symmetry=None)
    e1 = fit_generator_filter(f1, Zc, dt=dt, tikhonov_reg=0, rank=None, centered=True)
    k1 = modes(eig_result=e1, initial_conditions=Zc - fmean, obs_train=obs)
    r_to.append(resolvent_response(shift + 1j * th * 2 * np.pi, k1))
    f2 = GeneratorResolvent(n, n // 2, dt, shift=shift + 1j * th * 2 * np.pi, correction=True)
    e2 = fit_generator_filter(f2, Zc, dt=dt, tikhonov_reg=0, rank=20, centered=True, numerical_solver=qr_solve_hpd)
    k2 = modes(eig_result=e2, initial_conditions=Zc - fmean, obs_train=obs)
    r_gen.append(resolvent_response(shift + 1j * th * 2 * np.pi, k2))
print(f"chaotic filters done {time.time()-t0:.1f}s", flush=True)
np.savez(out / "chaotic.npz", thetas=thetas, r_koop=r_koop, r_to=np.array(r_to), r_gen=np.array(r_gen),
         clean=data[:ntr, -2:], noisy=train[:, -2:], full=sol)
print("done", flush=True)
