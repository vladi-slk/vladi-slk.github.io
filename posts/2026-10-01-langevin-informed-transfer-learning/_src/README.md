Figure sources for the LITL post (ignored by Quarto because the folder starts with `_`).

    pip install numpy scipy matplotlib
    python make_figs.py          # writes ../figs/litl-{hero,focusing,doublewell-table,chignolin-occupancy,fairness-ablation}.png

- `dw.py`: 1D double well, biased Langevin samples, LITL step 2 with a fixed Gaussian-RBF dictionary
  (importance weights e^{beta V}), finite-difference reference spectrum, projected drift.
- `sphere.py`: cobalt-nanoparticle generator on S^2. `reference()` solves the Witten-Laplacian form in
  real spherical harmonics (reproduces the paper's lambda_1 = -1.302, lambda_{2,3} = -2.48 at Ku=1, Kc=0.3,
  beta=1); `litl()` is the same estimator the browser widget (`../magnet-widget.js`) runs.
- Double-well, chignolin and fairness charts re-plot numbers from Tables 1-2 and Sec. 5 of the paper.
  `figs/litl-{chignolin-fes,cobalt-paper,fairness-pareto-paper}.png` are crops of Figs. 1, 2 and 10.
