#!/usr/bin/env python3
"""Fig. 3: how much the fill rule alone moves channel-dropout accuracy (feature-coordinate experiment).

Both panels plot the paired subject-level change in balanced accuracy relative to zero-fill,
with 95% CIs from the subject bootstrap, against the fraction of electrodes dropped (10-50%),
for the three fixed encoders (train-only PCA for the reservoir; E2 output).

Panel (a): train-mean fill minus zero-fill. The two constant fills differ only by the shift
-mu/sigma of Eq. (noninv), so Proposition 1 predicts no change for a centered code; the measured
mean |mu|/sigma of each encoder (E5) is printed in the legend when e5_controls.json exists.
Panel (b): kNN fill minus zero-fill. kNN replaces the dropped block by an average over ten
training neighbours, so it changes the content of the block, not only its origin.

Grayscale-legible and color-blind safe: encoders differ by marker and line style as well as by
color; nothing depends on color alone. Sized for one IEEE column. Reads only aggregate JSON.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = os.environ.get("FIG_SUBJ_JSON", "outputs/aggregate/e2_trainonly_pca_fills.json")
J = json.load(open(ROOT / SRC))
E5 = ROOT / "outputs/aggregate/e5_controls.json"
CENT = json.load(open(E5))["centeredness_mean_abs_mu_over_sigma"] if E5.exists() else None
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/imported/fig_impute_subj.pdf"))

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
ENC = ["ERP-window", "Band-power", "Reservoir"]
STYLE = {"ERP-window": dict(color="#D55E00", marker="o", ls="-"),
         "Band-power": dict(color="#0072B2", marker="s", ls="--"),
         "Reservoir": dict(color="#009E73", marker="^", ls=":")}
DODGE = {"ERP-window": -1.4, "Band-power": 0.0, "Reservoir": 1.4}
LEVELS = [10, 20, 30, 40, 50]

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
for ax, fill, title in ((axes[0], "mean", "(a) train-mean $-$ zero fill"),
                        (axes[1], "knn", "(b) $k$NN $-$ zero fill")):
    ax.axhline(0, color="k", lw=0.6)
    for enc in ENC:
        r = J["fill_minus_zero"][enc][fill]
        m = [r[str(l)]["mean_diff"] for l in LEVELS]
        lo = [r[str(l)]["mean_diff"] - r[str(l)]["ci95"][0] for l in LEVELS]
        hi = [r[str(l)]["ci95"][1] - r[str(l)]["mean_diff"] for l in LEVELS]
        st = STYLE[enc]
        label = enc
        if CENT is not None:
            label += f" ($|\\mu|/\\sigma$ {CENT[enc]['mean']:.2f})"
        ax.errorbar([l + DODGE[enc] for l in LEVELS], m, yerr=[lo, hi], color=st["color"],
                    marker=st["marker"], ls=st["ls"], lw=0.9, ms=3.0, capsize=1.5, elinewidth=0.6,
                    mfc="white" if enc == "Reservoir" else st["color"], label=label)
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xticks(LEVELS)
    ax.set_xlim(5, 55)
    ax.set_xlabel("electrodes dropped (%)", labelpad=1)
    ax.tick_params(length=2, pad=1.5)
axes[0].set_ylabel("$\\Delta$ balanced accuracy", labelpad=1)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3 if CENT is None else 2, fontsize=5.6,
           frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.0)
fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0, 1, 0.80 if CENT is not None else 0.86))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
print(f"[fig] wrote {OUT}")
