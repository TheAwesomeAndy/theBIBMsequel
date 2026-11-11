#!/usr/bin/env python3
"""Fig. 4 (camera-ready).

(a) External check on DEAP (analysis/deap_replication.py): band-power balanced accuracy versus
    channels removed under zero, train-mean and kNN fill, with subject-level 95% CIs; the
    clean level and chance are marked.
(b) Training effect (analysis/experiment_eegnet_signal.py): EEGNet with and without train-time
    augmentation under the shared signal-level conditions (clean, 5 dB noise, +/-50 ms
    jitter, 30% and 50% electrodes removed), subject-level 95% CIs.
Fill rules / models differ by marker, line style and hatch, not color alone. Reads only
aggregate JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs/aggregate"
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/imported/fig_deap_aug.pdf"))
plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "hatch.linewidth": 0.35, "pdf.fonttype": 42,
                     "ps.fonttype": 42})


def main():
    deap = json.load(open(A / "deap_replication_v2.json"))["BandPower"]
    e3 = json.load(open(os.environ.get("E3_JSON", A / "e3_eegnet_signal.json")))["BA"]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6))          # one IEEE column, printed at 100%

    rates = [10, 30, 50]; keys = ["0.10", "0.30", "0.50"]
    sty = {"zero": ("--", "o", "zero fill"), "mean": ("-.", "s", "mean fill"), "knn": ("-", "^", "kNN fill")}
    for fill, (ls, mk, lab) in sty.items():
        v = np.array([deap[k][fill]["BA_mean_over_folds"] for k in keys])
        lo = np.array([deap[k][fill]["subject_ci95"][0] for k in keys])
        hi = np.array([deap[k][fill]["subject_ci95"][1] for k in keys])
        ax[0].errorbar(rates, v, yerr=[v - lo, hi - v], color="black", ls=ls, marker=mk, ms=3,
                       lw=0.9, capsize=1.5, elinewidth=0.6, mfc="white" if fill == "zero" else "black",
                       label=lab)
    clean = deap["clean"]["BA_mean_over_folds"]
    ax[0].axhline(clean, color="0.45", ls=(0, (1, 1)), lw=0.8)
    ax[0].text(54, clean + 0.004, f"clean {clean:.3f}", fontsize=5.5, color="0.3", ha="right")
    ax[0].axhline(0.5, color="0.7", ls=(0, (1, 1)), lw=0.7)
    ax[0].text(51, 0.503, "chance", fontsize=5.5, color="0.45", ha="right")
    ax[0].set_xlabel("channels removed (%)", labelpad=1); ax[0].set_ylabel("balanced accuracy", labelpad=1)
    ax[0].set_title("(a) DEAP: fill rule", fontsize=6.5, pad=3)
    ax[0].set_xticks(rates); ax[0].set_xlim(5, 55)
    ax[0].legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=2.0, borderaxespad=0.2)
    ax[0].grid(True, color="0.92", lw=0.6); ax[0].set_axisbelow(True)

    conds = [("remove_0.0", "clean"), ("amp_5dB", "5 dB"), ("jitter_50ms", r"$\pm$50 ms"),
             ("remove_0.3", "30%"), ("remove_0.5", "50%")]
    x = np.arange(len(conds)); w = 0.36
    for j, (m, face, hatch, lab) in enumerate((("EEGNet", "#e6e6e6", "....", "EEGNet"),
                                              ("EEGNet+aug", "#7a7a7a", "////", "EEGNet + aug."))):
        v = np.array([e3[c][m]["BA"] for c, _ in conds])
        lo = np.array([e3[c][m]["ci95"][0] for c, _ in conds]); hi = np.array([e3[c][m]["ci95"][1] for c, _ in conds])
        ax[1].bar(x + (j - 0.5) * w, v, w, yerr=[v - lo, hi - v], capsize=1.2, color=face,
                  edgecolor="black", lw=0.4, hatch=hatch, label=lab, error_kw={"elinewidth": 0.6})
    ax[1].axhline(1 / 3, color="0.5", ls=(0, (3, 2)), lw=0.7)
    ax[1].set_xticks(x); ax[1].set_xticklabels([c[1] for c in conds], fontsize=5.5)
    ax[1].set_ylim(0.30, 0.90)
    ax[1].set_title("(b) EEGNet: training effect", fontsize=6.5, pad=3)
    ax[1].legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2, borderaxespad=0.2,
                 handlelength=1.6, columnspacing=0.8)
    ax[1].grid(True, axis="y", color="0.92", lw=0.6); ax[1].set_axisbelow(True)

    for a in ax:
        a.tick_params(length=2, pad=1.5)
    fig.tight_layout(pad=0.25, w_pad=0.8)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
