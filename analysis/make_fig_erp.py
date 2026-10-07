#!/usr/bin/env python3
"""Fig. 2: trial-averaged affective ERP data (SHAPE cohort, grand averages over 211 subjects).

(a) Grand-average waveforms per condition at the most affect-discriminative channel (largest
    |emotional - neutral| mean amplitude in the late ERP-window feature, sample-index 450-800 ms,
    i.e. about 250-600 ms after stimulus onset; time axes are plotted relative to onset);
(b) spatiotemporal map of the emotional-minus-neutral difference over all 34 channels, with
    "-"/"+" markers so the sign survives grayscale printing;
(c) global field power (SD across channels) per condition.
Conditions differ by line style as well as color (grayscale-legible). Reads the restricted
SHAPE pickle locally; writes only the figure (group averages, no subject-level data).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis_subject_bootstrap as RB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS_RAW = 1024       # recording rate of the SHAPE epochs (1229 samples = 1.2 s)
ONSET_RAW = 205     # stimulus onset: end of the 200 ms baseline-corrected interval
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/imported/fig_erp.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
NAMES = {0: "negative", 1: "neutral", 2: "pleasant"}
STYLE = {0: ("#0072B2", "-"), 1: ("#7f7f7f", ":"), 2: ("#D55E00", "--")}


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]).ravel()
    # X_ds is every fourth sample of the 1024 Hz epoch from its first sample, which is 205 samples
    # (200 ms) before stimulus onset (verified: X_ds == raw[0:1024:4], z-scored per epoch and channel).
    t = (4 * np.arange(X.shape[1]) - ONSET_RAW) / FS_RAW * 1000       # ms relative to onset
    t_idx = np.arange(X.shape[1]) / RB.FS * 1000                     # sample-index ms used by the analysis code
    ga = {c: X[y == c].mean(0) for c in (0, 1, 2)}                    # (256, 34)
    lpp = (t_idx >= 450) & (t_idx <= 800)                              # third ERP-window feature
    lo, hi = t[lpp].min(), t[lpp].max()                                # about 250-600 ms after onset
    emo = (ga[0] + ga[2]) / 2
    ch = int(np.argmax(np.abs((emo - ga[1])[lpp].mean(0))))

    fig = plt.figure(figsize=(7.16, 1.45))                 # one row across both IEEE columns
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    for c in (0, 1, 2):
        col, ls = STYLE[c]
        ax0.plot(t, ga[c][:, ch], color=col, ls=ls, lw=1.1, label=NAMES[c])
    ax0.axvspan(lo, hi, color="0.9", zorder=0); ax0.axhline(0, color="0.6", lw=0.6)
    ax0.axvline(0, color="k", lw=0.6, ls="--")
    ax0.set_xlabel("time (ms)", labelpad=1); ax0.set_ylabel("amplitude (a.u.)", labelpad=1)
    ax0.set_title(f"(a) grand-average ERP, channel {ch}", fontsize=7, loc="left", pad=3)
    ax0.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=2.0, borderaxespad=0.2)
    ax0.grid(True, color="0.93", lw=0.5)

    ax1 = fig.add_subplot(gs[1])
    diff = (emo - ga[1]).T                                            # (34, 256)
    vmax = np.percentile(np.abs(diff), 98)
    im = ax1.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                    norm=colors.TwoSlopeNorm(0, -vmax, vmax), extent=[t[0], t[-1], 0, diff.shape[0]])
    ax1.axvline(lo, color="k", lw=0.5, ls=":"); ax1.axvline(hi, color="k", lw=0.5, ls=":")
    ax1.axvline(0, color="k", lw=0.6, ls="--")
    for tx, lab in ((-100, "$-$"), (560, "$+$")):          # sign of the difference, legible without color
        ax1.text(tx, 3.5, lab, ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.5))
    ax1.set_xlabel("time (ms)", labelpad=1); ax1.set_ylabel("channel", labelpad=1)
    ax1.set_title("(b) emotional $-$ neutral, all channels", fontsize=7, loc="left", pad=3)
    fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5, length=1.5, pad=1)

    ax2 = fig.add_subplot(gs[2])
    for c in (0, 1, 2):
        col, ls = STYLE[c]
        ax2.plot(t, ga[c].std(1), color=col, ls=ls, lw=1.1)
    ax2.axvspan(lo, hi, color="0.9", zorder=0)
    ax2.axvline(0, color="k", lw=0.6, ls="--")
    ax2.set_xlabel("time (ms)", labelpad=1); ax2.set_ylabel("GFP (a.u.)", labelpad=1)
    ax2.set_title("(c) global field power", fontsize=7, loc="left", pad=3)
    ax2.grid(True, color="0.93", lw=0.5)

    for a in (ax0, ax1, ax2):
        a.tick_params(length=2, pad=1.5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}; affect-discriminative channel = {ch}")


if __name__ == "__main__":
    main()
