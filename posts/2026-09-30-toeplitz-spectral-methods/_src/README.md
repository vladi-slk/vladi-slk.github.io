Figure sources for the Toeplitz post (ignored by Quarto because the folder starts with `_`).

    pip install numpy scipy matplotlib torch scikit-learn ipython "kooplearn<2"
    git clone https://github.com/vladi-slk/tklearn.git        # or set TKLEARN_SRC=/path/to/tklearn/src
    python simulate_long.py      # long chaotic + clean periodic Duffing runs (cached .npy)
    python run_duffing.py        # re-runs the paper's Duffing experiments with tklearn (cached .npz)
    python make_figs.py          # writes ../figs/toeplitz-*.png

`run_duffing.py` follows notebooks/DuffingOscilator.ipynb in tklearn; its forecast RMSEs
(Euclidean over x and y) reproduce the published 1.250 / 1.299 / 0.336 (rank 10) and
0.501 / 0.253 / 0.328 (rank 100). Figures use the "Avenir Next" font (macOS).
