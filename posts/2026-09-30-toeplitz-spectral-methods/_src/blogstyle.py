"""Shared figure style for the two blog posts."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#1f2328", "#57606a", "#8c959f", "#e6e8eb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY1, GREY2 = "#a3a9b0", "#5f6368"


def setup():
    mpl.rcParams.update({
        "font.family": "Avenir Next", "font.size": 10.5,
        "mathtext.fontset": "stix",
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
        "axes.titlesize": 11.5, "axes.titleweight": "demibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 5, "ytick.major.pad": 5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 110,
        "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
