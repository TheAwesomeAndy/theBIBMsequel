"""Shared figure style. Categorical slots in fixed order (validated palette, light mode).

Leaking scalings take slots 1 and 2; scalings that use no post-stimulus sample
take slots 3 to 5 and carry a hatch as a second channel for grayscale print.
"""
import matplotlib as mpl

SCALE = {
    "epoch_z": ("per-epoch z", "#2a78d6", ""),
    "post_z": ("post-stim. z", "#eb6834", ""),
    "pre_sd": ("pre-stim. SD", "#1baf7a", "////"),
    "none": ("unscaled", "#eda100", "...."),
    "global_sd": ("cohort SD", "#e87ba4", "xxxx"),
}
COND = [("negative", "#2a78d6", "-"), ("neutral", "#555555", ":"), ("pleasant", "#eb6834", "--")]


def apply():
    mpl.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "font.family": "serif",
    })
