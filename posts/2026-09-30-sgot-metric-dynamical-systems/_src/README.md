Figure sources for the SGOT post (ignored by Quarto because the folder starts with `_`).

    pip install numpy scipy matplotlib
    python make_figs.py          # writes ../figs/sgot-{hero,shifts,classification,ranks}.png

`sgot_core.py` is a minimal re-implementation of SGOT (linear kernel, primal RRR, η-weighted
eigenvalue + Grassmann cost, W1 via assignment). The classification charts re-plot numbers from
Tables 1, 5, 7 and 8 of the paper. `figs/sgot-fluid-barycenter.png` is Fig. 5 of the ICLR paper
(cropped from page 10 of the PDF at 5x). Figures use the "Avenir Next" font (macOS).
