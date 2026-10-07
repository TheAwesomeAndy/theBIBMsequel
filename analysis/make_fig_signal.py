#!/usr/bin/env python3
"""Journal figure: signal-level perturbations on the external cohort.

Reads outputs/aggregate/journal/j2_tcrzem_signal{suffix}.json and j3_tcrzem_eegnet{suffix}.json
(suffix from $FIG_EPOCH: "" for the conference-matched epoch, "_long" for the pre-specified
long epoch).
(a) recorded observation noise: subject-level BA (95% CI) when each test observation is
    re-averaged from 1, 4 or 16 of its single trials, and with all trials (clean, median 44);
(b) electrode removal as a zero trace versus spherical-spline repair from the retained
    electrodes, at 10, 30 and 50% removed (EEGNet: 30 and 50%).
Encoders differ by marker and line style (grayscale-legible). Chance = 1/3.
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
AGG = ROOT / "outputs" / "aggregate" / "journal"
SUF = os.environ.get("FIG_EPOCH", "")
OUT = Path(os.environ.get("FIG_OUT", ROOT / f"manuscript/figures/journal/fig_signal{SUF}.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
STY = {"Band-power": ("#E69F00", "s", "-"), "ERP-window": ("#0072B2", "o", "--"),
       "Reservoir": ("#009E73", "^", ":"), "EEGNet": ("#CC79A7", "D", "-."),
       "EEGNet+aug": ("#D55E00", "v", (0, (3, 1, 1, 1)))}


def get(d, cond, n):
    e = d["BA"][cond][n]
    return e["BA"], e["ci95"][0], e["ci95"][1]


def main():
    s = json.load(open(AGG / f"j2_tcrzem_signal{SUF}.json"))
    p3 = AGG / f"j3_tcrzem_eegnet{SUF}.json"
    if not p3.exists():                       # long-epoch EEGNet runs on the seed-42 partition
        p3 = AGG / f"j3_tcrzem_eegnet_s42{SUF}.json"
    e = json.load(open(p3)) if p3.exists() else None
    med = s["trials_per_observation"]["median"]
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.85), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.5]})
    xs = [1, 4, 16, med]
    conds = ["trials_1", "trials_4", "trials_16", "clean"]
    series = [(n, s) for n in ("Band-power", "ERP-window", "Reservoir")]
    if e is not None:
        series += [(n, e) for n in ("EEGNet", "EEGNet+aug")]
    for n, src in series:
        col, mk, ls = STY[n]
        v = np.array([get(src, c, n) for c in conds])
        ax[0].errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, marker=mk,
                       ls=ls, lw=1.0, ms=3, capsize=1.5, elinewidth=0.6, label=n)
    ax[0].set_xscale("log"); ax[0].set_xticks(xs); ax[0].set_xticklabels(["1", "4", "16", f"all ({med:.0f})"])
    ax[0].axhline(1 / 3, color="0.4", lw=0.6)
    ax[0].set_xlabel("trials averaged per test observation", labelpad=1); ax[0].set_ylabel("BA", labelpad=1)
    ax[0].set_title("(a) recorded observation noise", fontsize=7, loc="left", pad=3)
    ax[0].legend(frameon=False, fontsize=5.2, loc="upper left", ncol=1)

    groups = [("Band-power", s, ["0.1", "0.3", "0.5"]), ("ERP-window", s, ["0.1", "0.3", "0.5"]),
              ("Reservoir", s, ["0.1", "0.3", "0.5"])]
    if e is not None:
        groups += [("EEGNet", e, ["0.3", "0.5"]), ("EEGNet+aug", e, ["0.3", "0.5"])]
    x0 = 0
    ticks, labels = [], []
    for n, src, levels in groups:
        col, mk, _ = STY[n]
        for i, lv in enumerate(levels):
            xz, xsp = x0 + i * 0.9, x0 + i * 0.9 + 0.32
            for xx, cond, fc in ((xz, f"remove_{lv}", "white"), (xsp, f"spline_{lv}", col)):
                b, lo, hi = get(src, cond, n)
                ax[1].errorbar([xx], [b], yerr=[[b - lo], [hi - b]], fmt=mk, ms=3.2, mfc=fc, mec=col,
                               ecolor=col, elinewidth=0.6, capsize=1.2)
            ticks.append(x0 + i * 0.9 + 0.16); labels.append(f"{int(float(lv) * 100)}%")
        cb = get(src, "clean", n)[0]
        ax[1].hlines(cb, x0 - 0.2, x0 + (len(levels) - 1) * 0.9 + 0.5, color=col, lw=0.6, ls=":")
        ax[1].text(x0 + (len(levels) - 1) * 0.45 + 0.16, 0.307, n, ha="center", fontsize=5.2)
        x0 += len(levels) * 0.9 + 0.6
    ax[1].set_xticks(ticks); ax[1].set_xticklabels(labels, fontsize=5)
    ax[1].axhline(1 / 3, color="0.4", lw=0.6)
    ax[1].set_ylim(bottom=0.30)
    ax[1].set_ylabel("BA", labelpad=1)
    ax[1].set_title("(b) electrodes removed: zero trace (open) vs spline repair (filled); dotted = clean",
                    fontsize=6.5, loc="left", pad=3)
    for a in ax:
        a.grid(True, axis="y", color="0.93", lw=0.5); a.tick_params(length=2, pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
