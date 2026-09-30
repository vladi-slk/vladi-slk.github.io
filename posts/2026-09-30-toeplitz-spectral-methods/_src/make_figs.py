import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.signal import welch
from scipy.special import sici
from blogstyle import *

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figs"
setup()
dt = 0.1

# ------------------------------------------------------------------ hero: chaotic attractor + its spectrum
X = np.load(HERE / "chaotic_long.npy").astype(float)
fig = plt.figure(figsize=(12, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.18)
ax = fig.add_subplot(gs[0])
seg = X[:30000]
tt = np.arange(len(seg)) * dt
pts = seg.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
cmap = LinearSegmentedColormap.from_list("ph", [BLUE, AQUA, YELLOW, ORANGE, MAGENTA, VIOLET, BLUE])
lc = LineCollection(segs, cmap=cmap, lw=0.35, alpha=0.55)
lc.set_array(np.mod(tt[:-1], 2 * np.pi))
ax.add_collection(lc)
ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.35, 1.35)
ax.set_xlabel("position  $x$"); ax.set_ylabel("velocity  $y=\\dot x$")
ax.set_title("Forced Duffing oscillator in its chaotic regime")
ax.grid(False)
ax.text(0.02, 0.02, "colour = phase of the periodic forcing", transform=ax.transAxes, fontsize=8.8, color=INK2)
ax2 = fig.add_subplot(gs[1])
y = X[:, 1] - X[:, 1].mean()
f, P = welch(y, fs=1 / dt, nperseg=2 ** 15, noverlap=2 ** 14, detrend=False)
w = 2 * np.pi * f
m = w <= 6
ax2.fill_between(w[m], 1e-7, P[m], color=BLUE, alpha=0.10, lw=0)
ax2.semilogy(w[m], P[m], color=BLUE, lw=1.3)
ax2.set_ylim(1e-5, 3e2); ax2.set_xlim(0, 6)
for k, lab in [(1, "forcing  ω = 1"), (3, "3ω"), (5, "5ω")]:
    i = np.argmin(np.abs(w - k)); ax2.annotate(lab, (w[i], P[i]), xytext=(8, 2), textcoords="offset points",
                                               fontsize=9.3, color=INK)
ax2.annotate("broadband, continuous part:\nno eigenvalues to find here", (2.0, 0.2), xytext=(3.2, 8e0),
             fontsize=9.3, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax2.set_xlabel("angular frequency  ω  (rad/s)"); ax2.set_ylabel("power spectral density of $y$")
ax2.set_title("Its velocity spectrum: sharp lines on a continuous background")
save(fig, OUT / "toeplitz-hero.png")

# ------------------------------------------------------------------ shift = Toeplitz: Z . T_n = Z_F
per = np.load(HERE / "periodic.npz")
noisy = per["noisy"]
n_show, m_rows, ell = 64, 14, 8
from numpy.lib.stride_tricks import sliding_window_view
Zx = sliding_window_view(noisy[:, 0], 10)[:n_show + 40]                         # delay coords of x
Z = np.concatenate([Zx, Zx ** 2 - (Zx ** 2).mean()], axis=1)[:, :m_rows].T    # (m, n)
Z = Z[:, :n_show]
j = np.arange(1, ell + 1)
fmin, fmax = 0.1, 1.0
a = -(sici(j * 2 * np.pi * dt * fmax)[0] - sici(j * 2 * np.pi * dt * fmin)[0]) / np.pi
a *= 1 - j / (ell + 1)
Tn = np.zeros((n_show, n_show))
for k, ak in zip(j, a):
    Tn += np.diag(np.full(n_show - k, ak), k) - np.diag(np.full(n_show - k, ak), -k)
ZF = Z @ Tn
fig, axs = plt.subplots(1, 5, figsize=(12, 3.3), gridspec_kw=dict(width_ratios=[1, 0.12, 1, 0.12, 1]))
norm = lambda M: TwoSlopeNorm(0, -np.abs(M).max(), np.abs(M).max())
div = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
for ax_, M, ttl in [(axs[0], Z, "time-ordered features  $Z$\n(rows: features, columns: $t_1,\\dots,t_n$)"),
                    (axs[2], Tn, "banded Toeplitz matrix  $T_n$\n(constant along each diagonal)"),
                    (axs[4], ZF, "filtered features  $Z\\,T_n$\n(one matrix product = a spectral filter)")]:
    ax_.imshow(M, cmap=div, norm=norm(M), aspect="auto", interpolation="nearest")
    ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
    for s_ in ax_.spines.values():
        s_.set_visible(False)
    ax_.set_title(ttl, fontsize=10, loc="center", fontweight="normal", color=INK)
for ax_, sym in [(axs[1], "×"), (axs[3], "=")]:
    ax_.axis("off"); ax_.text(0.5, 0.5, sym, fontsize=26, ha="center", va="center", color=INK2)
axs[2].set_aspect("equal")
save(fig, OUT / "toeplitz-identity.png")

# ------------------------------------------------------------------ filter gallery (deterministic: spectrum on iR)
om = np.linspace(0.05, np.pi / (2 * dt), 4000)
z = np.exp(1j * om * dt)
def lap(mu, ell):
    jj = np.arange(ell + 1); c = dt * np.exp(-mu * jj * dt); c[0] *= .5; c[-1] *= .5
    return (c[None, :] * z[:, None] ** jj[None, :]).sum(1)
def band(ell, fmin=0.1, fmax=1.0):
    jj = np.arange(1, ell + 1)
    c = -(sici(jj * 2 * np.pi * dt * fmax)[0] - sici(jj * 2 * np.pi * dt * fmin)[0]) / np.pi
    c *= 1 - jj / (ell + 1)
    return (c[None, :] * (z[:, None] ** jj - z[:, None] ** (-jj))).sum(1)
mu_res = 0.25 + 3j  # resolvent (mu - L)^{-1}, zoom at omega = 3 rad/s
panels = [
    ("Koopman  $e^{\\Delta t L}$", np.abs(z), np.abs(z), "every harmonic equally loud: the filter does not choose"),
    ("hyperbolic sine  $\\sinh(\\Delta t L)$", np.abs(np.sin(om * dt)), np.abs((z - 1 / z) / 2), "high frequencies dominate; imaginary spectrum by design"),
    ("resolvent  $(\\mu-L)^{-1}$", 1 / np.abs(mu_res - 1j * om), np.abs(lap(mu_res, 400)), "a zoom lens: modes near Im μ = 3 rad/s dominate"),
    ("band-limited  $P_{[f_1,f_2]}L^{-1}$", np.where((om >= 2 * np.pi * .1) & (om <= 2 * np.pi), 1 / (om * dt), 0),
     np.abs(band(300)), "keeps 0.1–1 Hz; the slowest modes in the band dominate"),
]
cols = [GREY2, BLUE, AQUA, ORANGE]
fig, axs = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
kk = np.arange(1, 16)
for ax_, (ttl, ideal, real, note), c in zip(axs.flat, panels, cols):
    sc = ideal.max() if ideal.max() > 0 else 1
    ax_.semilogx(om, ideal / sc, color=GRID, lw=5, solid_capstyle="butt")
    ax_.semilogx(om, np.abs(real) / sc, color=c, lw=1.8)
    idx = [np.argmin(np.abs(om - k)) for k in kk]
    ax_.scatter(kk, np.abs(real[idx]) / sc, s=16, color=INK, zorder=4, lw=0)
    ax_.set_title(ttl, fontsize=11, loc="left", pad=22)
    ax_.text(0, 1.04, note, transform=ax_.transAxes, fontsize=9.3, color=INK2, va="bottom")
    ax_.set_xlim(0.3, np.pi / (2 * dt)); ax_.set_ylim(-0.03, 1.12)
    ax_.set_xticks([1, 3, 10]); ax_.set_xticklabels(["1", "3", "10"])
for ax_ in axs[1]:
    ax_.set_xlabel("ω (rad/s)", fontsize=10)
for ax_ in axs[:, 0]:
    ax_.set_ylabel("|F(iω)|  (normalised)")
fig.tight_layout(h_pad=2.2, w_pad=1.5)
save(fig, OUT / "toeplitz-filters.png")

# ------------------------------------------------------------------ Gibbs vs damping
phi = np.linspace(1e-3, 0.4 * np.pi, 5000)
fz = phi / (2 * np.pi * dt)
zz = np.exp(1j * phi)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for ax_, ell in zip(axs, [15, 60, 240]):
    jj = np.arange(1, ell + 1)
    c = -(sici(jj * 2 * np.pi * dt * 1.0)[0] - sici(jj * 2 * np.pi * dt * 0.1)[0]) / np.pi
    raw = np.abs((c[None, :] * (zz[:, None] ** jj - zz[:, None] ** (-jj))).sum(1))
    cd = c * (1 - jj / (ell + 1))
    damp = np.abs((cd[None, :] * (zz[:, None] ** jj - zz[:, None] ** (-jj))).sum(1))
    ideal = np.where((fz >= .1) & (fz <= 1), 1 / phi, 0)
    ax_.plot(fz, ideal, color=GRID, lw=5, solid_capstyle="butt", label="ideal symbol")
    ax_.plot(fz, raw, color=GREY1, lw=1.4, label="truncated (Gibbs)")
    ax_.plot(fz, damp, color=ORANGE, lw=2, label="damped coefficients")
    ax_.set_title(f"ℓ = {ell} Toeplitz diagonals", fontsize=10.5)
    ax_.set_xlabel("frequency (Hz)"); ax_.set_xlim(0, 2)
axs[0].set_ylabel("$|T_\\ell(e^{i\\omega\\Delta t})|$")
axs[0].legend(loc="upper right", fontsize=9)
axs[0].set_ylim(0, 19)
save(fig, OUT / "toeplitz-gibbs.png")

# ------------------------------------------------------------------ periodic regime: forecasts
P_ = per["preds"]; truth = per["truth"]
tgrid = np.arange(truth.shape[0]) * dt
names = ["Koopman  $e^{\\Delta t L}$  (Hankel-DMD / RRR)", "Hyperbolic sine  $\\sinh(\\Delta t L)$",
         "Band-limited inverse  $P_{[0.1,1]\\,\\mathrm{Hz}}L_0^{-1}$"]
cols = [GREY2, BLUE, ORANGE]
fig, axs = plt.subplots(3, 2, figsize=(12, 6.2), sharex=True, sharey=True)
for mi in range(3):
    for ci, ri in enumerate([1, 2]):
        ax_ = axs[mi, ci]
        pr = P_[mi, ri]
        mean = pr.mean(0); lo, hi = np.percentile(pr, [5, 95], axis=0)
        rmse = np.sqrt(((mean - truth) ** 2).sum(1).mean())
        ax_.plot(tgrid, truth[:, 0], color=INK, lw=1.0, label="true trajectory")
        ax_.fill_between(tgrid, lo[:, 0], hi[:, 0], color=cols[mi], alpha=0.18, lw=0)
        ax_.plot(tgrid, mean[:, 0], color=cols[mi], lw=2.0, label="forecast (mean of 10 noisy trainings)")
        ax_.text(0.99, 0.96, f"RMSE {rmse:.3f}", transform=ax_.transAxes, ha="right", va="top", fontsize=9.3,
                 color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
        ax_.set_ylim(-2.1, 2.3)
        if mi == 0:
            ax_.set_title(("rank 10" if ri == 1 else "rank 100 (no rank reduction)"), loc="center", fontsize=11)
        if ci == 0:
            ax_.set_ylabel("$x(t)$")
        ax_.text(0.01, 0.96, names[mi], transform=ax_.transAxes, fontsize=9.3, va="top", color=INK,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
for ax_ in axs[-1]:
    ax_.set_xlabel("forecast horizon (s)")
save(fig, OUT / "toeplitz-forecast.png")

# ------------------------------------------------------------------ periodic regime: data + eigenvalues
clean = np.load(HERE / "periodic_clean.npy").astype(float)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax_ = axs[0]
ax_.scatter(noisy[:, 0], noisy[:, 1], s=2, color=BLUE, alpha=0.25, lw=0, label="noisy training samples (σ = 0.3)")
ax_.plot(clean[:700, 0], clean[:700, 1], color=INK, lw=1.4, label="true limit cycle")
ax_.set_xlabel("$x$"); ax_.set_ylabel("$y=\\dot x$"); ax_.set_aspect("equal")
ax_.set_title("What the estimators see: 8,000 noisy samples")
ax_.legend(loc="upper left", fontsize=8.8, markerscale=4)
ax_ = axs[1]
ax_.axvline(0, color=MUTED, lw=1)
for k in range(-4, 5):
    ax_.axhline(k, color=GRID, lw=0.8)
for mi, (mk, lab) in enumerate([("s", "Koopman"), ("o", "hyperbolic sine"), ("*", "band-limited inverse")]):
    ev, wt = per[f"ev{mi}"], per[f"w{mi}"]
    s_ = 30 + 220 * wt / wt.max()
    ax_.scatter(ev.real, ev.imag, s=s_, marker=mk, color=cols[mi], alpha=0.85, edgecolor="white", lw=0.8,
                label=lab, zorder=3 + mi)
ax_.set_xlim(-1.6, 0.35); ax_.set_ylim(-4.6, 4.6)
ax_.set_xlabel("Re λ  (decay, 1/s)"); ax_.set_ylabel("Im λ  (rad/s)")
ax_.set_title("Estimated generator eigenvalues (marker size = mode's weight)")
ax_.text(-1.55, 4.15, "true spectrum:  λ = ik,  k = 0, ±1, ±2, …", fontsize=9, color=INK2)
ax_.legend(loc="lower left", fontsize=9)
save(fig, OUT / "toeplitz-periodic-setup.png")

# ------------------------------------------------------------------ chaotic regime: resolvent response
ch = np.load(HERE / "chaotic.npz")
th = ch["thetas"] * 2 * np.pi
fig, ax_ = plt.subplots(figsize=(10, 4.2))
for key, lab, c, lw in [("r_koop", "eigen-decomposition of a Koopman estimate (Hankel-DMD)", GREY1, 1.8),
                        ("r_to", "Toeplitz filter: Koopman resolvent  $(e^{\\mu+i\\omega}-e^{\\Delta t L})^{-1}$", BLUE, 1.8),
                        ("r_gen", "Toeplitz filter: generator resolvent  $(\\mu+i\\omega-L)^{-1}$", ORANGE, 2.2)]:
    r = ch[key] / ch[key].max()
    ax_.plot(th, r, color=c, lw=lw, label=lab)
for k in [-1, 1]:
    ax_.annotate("forcing\nfrequency", (k, 1.0), xytext=(k * 1.9, 0.93), fontsize=9, color=INK2, ha="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax_.set_xlim(-5, 5); ax_.set_ylim(-0.02, 1.08)
ax_.set_xlabel("ω  (rad/s)"); ax_.set_ylabel("$\\|(\\mu+i\\omega-L)^{-1}y\\|$  (normalised)")
ax_.set_title("Resolvent response of the velocity on the strange attractor (μ = 0.01)")
ax_.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, -0.18), ncol=1)
save(fig, OUT / "toeplitz-chaotic-resolvent.png")
