import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from blogstyle import *
from sgot_core import *

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figs"
setup()
fs = 200

# ------------------------------------------------------------------ Fig: metric behaviour under shifts
_, s0 = oscillators([0.5, 1.0], [0, 0], fs, seed=0)
ref = rrr(s0, fs, 4)
freqs = np.linspace(0.6, 2.5, 39)
decs = np.linspace(-0.3, 3.0, 67)
res = {"freq": [], "dec": []}
for i, f in enumerate(freqs):
    _, s = oscillators([0.5, f], [0, 0], fs, seed=100 + i)
    res["freq"].append(all_metrics(ref, rrr(s, fs, 4)))
for i, d in enumerate(decs):
    _, s = oscillators([0.5, 1.0], [0, d], fs, seed=300 + i)
    res["dec"].append(all_metrics(ref, rrr(s, fs, 4)))

series = [("hs", "Hilbert–Schmidt norm", GREY1, 1.6),
          ("op", "Operator norm", GREY2, 1.6),
          ("eig", "Eigenvalues only", BLUE, 1.8),
          ("sub", "Eigenspaces only", AQUA, 1.8),
          ("sgot", "SGOT", ORANGE, 2.8)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
for ax, key, xs, xl, x0, ttl in [(axes[0], "freq", freqs, "frequency of the 2nd oscillator (Hz)", 1.0,
                                  "(a) Shift the frequency"),
                                 (axes[1], "dec", decs, "decay rate of the 2nd oscillator (1/s)", 0.0,
                                  "(b) Shift the damping")]:
    for k, lab, c, lw in series:
        v = np.array([r[k] for r in res[key]])
        v = v / v.max()
        ax.plot(xs, v, color=c, lw=lw, label=lab, zorder=5 if k == "sgot" else 3)
    ax.axvline(x0, color=MUTED, lw=1, zorder=1)
    ax.text(x0, 1.07, " reference system", color=INK2, fontsize=9, va="bottom", ha="left" if key == "dec" else "center")
    ax.set_xlabel(xl)
    ax.set_title(ttl)
    ax.set_ylim(-0.03, 1.12)
axes[0].set_ylabel("distance to reference\n(normalised by max)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), handlelength=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, OUT / "sgot-shifts.png")

# ------------------------------------------------------------------ Hero: two systems as clouds of spectral atoms
fA, dA, aA = [0.45, 1.05, 1.9], [0.05, 0.25, 0.6], [1.0, 0.7, 0.5]
fB, dB, aB = [0.6, 1.35, 1.75], [0.15, 0.08, 0.35], [1.0, 0.6, 0.6]
def damped_signal(f, d, a, T=30, seed=0):
    t = np.arange(int(T * fs) + 1) / fs
    s = sum(ai * np.exp(-di * t) * np.sin(2 * np.pi * fi * t) for fi, di, ai in zip(f, d, a))
    return t, s + 1e-3 * np.random.default_rng(seed).standard_normal(len(t))
tA, sA = damped_signal(fA, dA, aA, T=12, seed=1)
tB, sB = damped_signal(fB, dB, aB, T=12, seed=2)
opA, opB = rrr(sA, fs, 6), rrr(sB, fs, 6)
C, _, _ = cost_matrix(opA, opB, 0.5)
r_idx, c_idx = linear_sum_assignment(C)

fig = plt.figure(figsize=(12, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], wspace=0.12, hspace=0.35)
for row, (t, s, c, name) in enumerate([(tA, sA, BLUE, "System A"), (tB, sB, ORANGE, "System B")]):
    ax = fig.add_subplot(gs[row, 0])
    m = t <= 8
    ax.plot(t[m], s[m], color=c, lw=1.8)
    ax.set_yticks([]); ax.set_xlim(0, 8)
    ax.grid(False); ax.spines["left"].set_visible(False)
    ax.set_title(f"{name}: observed trajectory", fontsize=10.5)
    if row == 1:
        ax.set_xlabel("time (s)")
    else:
        ax.set_xticklabels([])
axS = fig.add_subplot(gs[:, 1])
tAa, fAa, _, xA = atoms(opA)
tBa, fBa, _, xB = atoms(opB)
for i, j in zip(r_idx, c_idx):
    if fAa[i] > 0:
        axS.annotate("", xy=(fBa[j], -tBa[j]), xytext=(fAa[i], -tAa[i]),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=15, shrinkB=15,
                                     mutation_scale=13, connectionstyle="arc3,rad=0.2"))
for tt, ff, xx, c in [(tAa, fAa, xA, BLUE), (tBa, fBa, xB, ORANGE)]:
    for k in range(len(tt)):
        if ff[k] <= 0:
            continue
        x, y = ff[k], -tt[k]
        axS.scatter([x], [y], s=1100, color=c, alpha=0.12, lw=0, zorder=3)
        axS.scatter([x], [y], s=60, color=c, edgecolor="white", lw=2, zorder=4)
        ins = axS.inset_axes([x + 0.07, y - 0.035, 0.2, 0.07], transform=axS.transData)
        g = xx[:, k]
        g = (g * np.exp(-1j * np.angle(g[np.argmax(np.abs(g))]))).real
        ins.plot(np.linspace(0, 1, len(g)), g / np.abs(g).max(), color=c, lw=1.5)
        ins.set_ylim(-1.25, 1.25); ins.axis("off")
axS.set_xlim(0.25, 2.25); axS.set_ylim(-0.06, 0.72)
axS.invert_yaxis()
axS.set_xlabel("frequency  Im(λ)/2π  (Hz)")
axS.set_ylabel("damping  −Re(λ)  (1/s)")
axS.set_title("Spectral atoms: eigenvalue (dot) + eigenspace (wave glyph), and the optimal transport plan")
axS.scatter([], [], s=60, color=BLUE, label="system A")
axS.scatter([], [], s=60, color=ORANGE, label="system B")
axS.plot([], [], color=INK2, lw=1.4, label="optimal transport")
axS.legend(loc="lower left", fontsize=9.3)
save(fig, OUT / "sgot-hero.png")
print("A lam", np.round(opA["lam"], 3)); print("B lam", np.round(opB["lam"], 3))

# ------------------------------------------------------------------ Classification (numbers from Table 5 / Tables 1,7,8 of the paper)
names = ["AtrialFibrillation", "BasicMotions", "Cricket", "ERing", "EigenWorms", "Epilepsy", "FingerMovements",
         "HandMovementDirection", "Handwriting", "Heartbeat", "NATOPS", "SelfRegulationSCP1", "StandWalkJump",
         "UWaveGestureLibrary"]
cols = ["Hilbert–Schmidt", "Operator", "Martin", "SOT", "GOT", "SGOT"]
N = np.nan
acc = np.array([
    [0.31, 0.32, 0.27, 0.24, 0.40, 0.44], [0.48, 0.51, 0.30, 0.35, 0.80, 0.93], [0.33, 0.28, 0.07, 0.11, 0.63, 0.85],
    [0.79, 0.74, 0.15, 0.39, 0.85, 0.87], [0.60, 0.57, N, 0.57, 0.71, 0.88], [0.46, 0.52, N, 0.34, 0.78, 0.93],
    [0.51, 0.54, N, 0.51, 0.51, 0.57], [0.23, 0.23, 0.27, 0.21, 0.24, 0.29], [0.12, 0.12, 0.05, 0.05, 0.21, 0.42],
    [0.70, N, 0.71, 0.69, 0.70, 0.73], [0.76, 0.73, 0.25, 0.41, 0.74, 0.77], [0.57, 0.56, N, 0.57, 0.56, 0.61],
    [0.50, 0.41, N, 0.39, 0.30, 0.69], [0.24, 0.21, N, 0.13, 0.47, 0.64]])
best_other = np.nanmax(acc[:, :5], axis=1)
best_name = [cols[int(np.nanargmax(r[:5]))] for r in acc]
order = np.argsort(acc[:, 5] - best_other)
fig, ax = plt.subplots(figsize=(8.2, 5.6))
y = np.arange(len(names))
for yi, i in enumerate(order):
    ax.plot([best_other[i], acc[i, 5]], [yi, yi], color=GRID, lw=3.2, zorder=1, solid_capstyle="round")
    ax.scatter(best_other[i], yi, s=58, color=GREY2, edgecolor="white", lw=2, zorder=3)
    ax.scatter(acc[i, 5], yi, s=70, color=ORANGE, edgecolor="white", lw=2, zorder=4)
    ax.text(acc[i, 5] + 0.018, yi, f"+{100*(acc[i,5]-best_other[i]):.0f} pts", va="center", fontsize=8.8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=9.3)
ax.set_xlim(0.1, 1.04); ax.set_xlabel("k-NN classification accuracy (mean over 10 Monte-Carlo splits)")
ax.grid(axis="y", visible=False)
ax.scatter([], [], s=58, color=GREY2, label="best of 5 competing distances")
ax.scatter([], [], s=70, color=ORANGE, label="SGOT")
fig.legend(loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.55, -0.045))
ax.set_title("SGOT vs. the best competitor on 14 UEA multivariate time-series datasets (linear kernel)")
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, OUT / "sgot-classification.png")

ranks = {"Linear kernel": [3.29, 3.92, 5.30, 4.49, 2.66, 1.34],
         "Gaussian (RBF) kernel": [3.74, N, 4.02, 3.28, 2.48, 1.48],
         "Learned deep features": [3.33, 4.14, 5.06, 3.84, 2.94, 1.71]}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
for ax, (k, v) in zip(axes, ranks.items()):
    v = np.array(v)
    for j, (c, val) in enumerate(zip(cols, v)):
        if np.isnan(val):
            ax.text(0.1, j, "n/a", va="center", fontsize=9, color=MUTED)
            continue
        ax.barh(j, val, height=0.55, color=ORANGE if c == "SGOT" else "#c9ced4", zorder=2)
        ax.text(val + 0.08, j, f"{val:.2f}", va="center", fontsize=9, color=INK if c == "SGOT" else INK2,
                fontweight="demibold" if c == "SGOT" else "normal")
    ax.set_title(k, fontsize=10.5)
    ax.set_xlim(0, 6.2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("average rank (lower is better)")
axes[0].set_yticks(range(6)); axes[0].set_yticklabels(cols)
axes[0].invert_yaxis()
fig.tight_layout()
save(fig, OUT / "sgot-ranks.png")
