"""Figures for the LITL post. Run: python make_figs.py  (writes ../figs/*.png)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib.pyplot as plt
from blogstyle import *
import dw
from dw import U, dU, V, dV, beta, reference, simulate, litl_1d, psi_eval, drift

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figs"
setup()

# ------------------------------------------------------------------ hero: transfer from biased samples (1D double well)
xg, lam_t, psi_t = reference(U)
_, lam_b, psi_b = reference(lambda x: U(x) + V(x))
xs = simulate(lambda x: dU(x) + dV(x), 40000, seed=1)          # biased (source) samples only
m_litl = litl_1d(xs, beta * V(xs))                                # importance weights e^{beta V}
m_naive = litl_1d(xs, 0 * xs)                                      # same estimator, no reweighting
print("ref", lam_t[:4], "litl", m_litl["lam"][:4], "naive", m_naive["lam"][:4], "biased ref", lam_b[:4])

def align(f, ref):
    return f * np.sign((f * ref).sum())

x = np.linspace(-1.45, 1.45, 400)
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4))
ax = axs[0, 0]
ax.hist(xs, bins=80, range=(-1.6, 1.6), density=True, color=BLUE, alpha=0.22, lw=0, label="biased samples (what we have)")
pt = np.exp(-beta * U(x)); pt /= np.trapezoid(pt, x)
ax.plot(x, pt, color=INK, lw=1.4, label="target Boltzmann density")
ax2 = ax.twinx()
ax2.plot(x, U(x), color=GREY2, lw=1.6, ls="--", label="target potential U")
ax2.plot(x, U(x) + V(x), color=BLUE, lw=1.8, ls="--", label="biased potential U + V")
ax2.set_ylim(-1, 12); ax2.set_yticks([]); ax2.grid(False)
for s_ in ["right", "top"]:
    ax2.spines[s_].set_visible(False)
ax.set_ylim(0, 1.6); ax.set_yticks([])
ax.set_title("(a) The data come from the wrong system")
ax.set_xlabel("x")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.6, ncol=2, bbox_to_anchor=(0.5, -0.17))

ax = axs[0, 1]
ref1 = np.interp(x, xg, psi_t[:, 1]); refb = np.interp(x, xg, psi_b[:, 1])
ax.plot(x, ref1, color=GRID, lw=6, label=r"target $\psi_1$ (exact)")
ax.plot(x, align(psi_eval(m_litl, x, 1), ref1), color=ORANGE, lw=2.2, label="LITL from biased samples")
ax.plot(x, align(psi_eval(m_naive, x, 1), ref1), color=BLUE, lw=1.6, ls="--", label="same estimator, no reweighting")
ax.set_title("(b) The slowest eigenfunction: who is in which well")
ax.set_xlabel("x"); ax.legend(loc="lower right", fontsize=8.8)

ax = axs[1, 0]
ax.plot(x, dU(x), color=GRID, lw=6, label=r"true force $U'(x)$")
for r, c, a in [(2, GREY1, 1), (4, BLUE, 1), (len(m_litl["lam"]) - 1, ORANGE, 1)]:
    lab = f"LITL, rank {r}" if r < len(m_litl["lam"]) - 1 else f"LITL, all {r} modes"
    ax.plot(x, drift(m_litl, x, r), color=c, lw=2 if c == ORANGE else 1.5, label=lab)
ax.set_ylim(-25, 25)
ax.set_title("(c) Projected drift learned from energy differences")
ax.set_xlabel("x"); ax.legend(loc="lower right", fontsize=8.8)

ax = axs[1, 1]
taus = [1 / abs(lam_t[1]), 1 / abs(m_litl["lam"][1]), 1 / abs(m_naive["lam"][1]), 1 / abs(lam_b[1])]
labs = ["target system\n(exact)", "LITL\n(biased data)", "no reweighting\n(biased data)", "biased system\n(exact)"]
cols = [INK, ORANGE, BLUE, "#9ec5f4"]
ax.barh(range(4), taus, color=cols, height=0.55, zorder=2)
for i, t in enumerate(taus):
    ax.text(t * 1.08, i, f"{t:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(0.2, 40)
ax.set_yticks(range(4)); ax.set_yticklabels(labs, fontsize=9.2); ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel(r"slowest timescale  $1/|\lambda_1|$  (log scale)")
ax.set_title("(d) Well-to-well switching time")
fig.tight_layout(h_pad=2.6, w_pad=2.0)
save(fig, OUT / "litl-hero.png")

# ------------------------------------------------------------------ spectral focusing factor
fig, ax = plt.subplots(figsize=(9, 3.0))
r = np.logspace(-2, 2, 400)
f = (np.sqrt(r) + 1 / np.sqrt(r)) ** 2
ax.semilogx(r, f, color=ORANGE, lw=2.2)
ax.scatter([1], [4], s=50, color=ORANGE, edgecolor="white", lw=2, zorder=4)
ax.annotate(r"minimum 4 at $\mu = |\lambda_i|$", (1, 4), xytext=(2.2, 18), fontsize=9.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.set_xticks([0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"]); ax.set_ylim(0, 60); ax.set_xlabel(r"shift relative to the eigenvalue you want,  $\mu/|\lambda_i|$")
ax.set_ylabel("error amplification")
ax.set_title(r"The shift $\mu$ is a focusing knob: eigenvalue error is amplified by $(\sqrt{\mu/|\lambda_i|}+\sqrt{|\lambda_i|/\mu})^2$")
save(fig, OUT / "litl-focusing.png")

# ------------------------------------------------------------------ double-well table (paper Table 1)
true = np.array([0.22, 15.37, 16.06, 47.31])
meth = {"Devergne et al. 2024": [0.621, 46.6, 49.1, 146], "Zhang, Li & Schütte 2022": [1, 2, 2, 8],
        "LITL": [0.23, 15.315, 16.09, 46.9]}
cols = {"Devergne et al. 2024": GREY2, "Zhang, Li & Schütte 2022": BLUE, "LITL": ORANGE}
fig, ax = plt.subplots(figsize=(9, 3.4))
for j, (k, v) in enumerate(meth.items()):
    err = np.abs(np.array(v) - true) / true * 100
    ax.scatter(np.arange(4) + (j - 1) * 0.16, err, s=70 if k == "LITL" else 55, color=cols[k], edgecolor="white", lw=2,
               zorder=3, label=k)
ax.set_yscale("log"); ax.set_ylim(0.05, 900)
ax.set_xticks(range(4)); ax.set_xticklabels([rf"$\lambda_{i+1}$ = {t:g}" for i, t in enumerate(true)])
ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1%", "1%", "10%", "100%"]); ax.axhline(100, color=MUTED, lw=1, zorder=1)
ax.set_ylabel("relative error"); ax.grid(axis="x", visible=False)
ax.set_title("Double-well benchmark: eigenvalues of the target generator from biased data")
ax.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.14))
save(fig, OUT / "litl-doublewell-table.png")

# ------------------------------------------------------------------ chignolin occupancies (paper Sec. 5)
fig, ax = plt.subplots(figsize=(9, 2.3))
occ = {"molecular dynamics": [77, 8, 15], "BioEmu samples": [60, 24, 16]}
cc = [ORANGE, YELLOW, BLUE]; names = ["folded", "intermediate", "unfolded"]
for i, (k, v) in enumerate(occ.items()):
    left = 0
    for j, val in enumerate(v):
        ax.barh(i, val - 0.4, left=left, color=cc[j], height=0.5)
        ax.text(left + val / 2, i, f"{val}%", ha="center", va="center", fontsize=9.5,
                color="white" if j != 1 else INK, fontweight="demibold")
        left += val
ax.set_yticks([0, 1]); ax.set_yticklabels(list(occ)); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xticks([]); ax.grid(False)
for s_ in ["left", "bottom"]:
    ax.spines[s_].set_visible(False)
ax.set_title("Metastable-state occupancies from the generator LITL learned on each data source")
for j in range(3):
    ax.barh([], [], color=cc[j], label=names[j])
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9.3)
save(fig, OUT / "litl-chignolin-occupancy.png")

# ------------------------------------------------------------------ fairness ablation (paper Table 2)
rows = [("LITL (full)", 97.11, 1.00, 94.16, 4.04, ORANGE, "o"),
        ("LITL, local finite-difference proxy", 99.91, 0.09, 8.02, 2.41, "#f3b597", "o"),
        ("LITL, no audit feedback", 99.27, 0.37, 59.88, 5.08, "#f3b597", "o"),
        ("LITL, wrong spectral truncation", 95.68, 0.89, 70.68, 5.72, "#f3b597", "o"),
        ("retraining (needs gradients)", 101.32, 0.21, 94.60, 4.30, GREY2, "s"),
        ("fine-tuning (needs gradients)", 100.25, 0.17, 90.11, 5.78, GREY2, "s"),
        ("zero-order latent correction", 98.34, 0.61, 32.33, 2.51, BLUE, "D")]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
offs = {"LITL (full)": (0, -1.35, "center"), "LITL, local finite-difference proxy": (0, 0.55, "left"),
        "LITL, no audit feedback": (0, 0.55, "center"), "LITL, wrong spectral truncation": (0, -1.25, "center"),
        "retraining (needs gradients)": (0, 0.45, "center"), "fine-tuning (needs gradients)": (0, -0.55, "center"),
        "zero-order latent correction": (0, -0.85, "center")}
for name, a, sa, g, sg, c, mk in rows:
    ax.errorbar(g, a, xerr=sg, yerr=sa, fmt=mk, color=c, ms=9, mec="white", mew=1.5, ecolor=c, elinewidth=1.2,
                capsize=0, zorder=3)
    dx, dy, ha = offs[name]
    ax.text(g + dx - (2 if ha == "left" else 0), a + dy, name, fontsize=9, color=INK, va="center", ha=ha,
            fontweight="demibold" if name == "LITL (full)" else "normal")
ax.set_xlim(0, 112); ax.set_ylim(94.0, 102.5)
ax.set_xlabel("TPR gap removed  (% of the original classifier's gap)")
ax.set_ylabel("accuracy kept  (%)")
ax.set_title("Adult Income: post-hoc fairness steering, 10 trials (paper Table 2)")
save(fig, OUT / "litl-fairness-ablation.png")
