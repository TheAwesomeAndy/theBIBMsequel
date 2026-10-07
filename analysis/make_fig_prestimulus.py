#!/usr/bin/env python3
"""Journal figure: decoding from the empty pre-stimulus interval depends on normalization.

Reads outputs/aggregate/journal/j6_tcrzem_prestimulus{,_long}.json: (a) conference-matched
epoch, (b) pre-specified long epoch (adds 0..+2484 ms windows). Subject-level BA (95% CI) for
every encoder window under per-epoch z-scoring (Z, conference convention; filled markers)
and one fixed per-channel scale (G; open markers). Windows before onset carry no
stimulus-locked signal, so accuracy above chance there measures what the normalization
moves into the baseline. Chance = 1/3.
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
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_prestimulus.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
ORDER = [("ERP-window:pre_only", "ERP $-$200..0"), ("Reservoir:pre", "Res. $-$188..0"),
         ("Reservoir:conference", "Res. $-$160..+73"), ("Reservoir:early_post", "Res. 0..+234"),
         ("Reservoir:post", "Res. 0..+797"), ("ERP-window:post_only", "ERP +50..600"),
         ("ERP-window:conference", "ERP conference")]


LONG_EXTRA = [("Reservoir:post_long", "Res. 0..+2484"), ("ERP-window:post_only_long", "ERP +50..2500")]


def panel(ax, d, order, title):
    x = np.arange(len(order))
    for off, nm, mk, fc, lab in ((-0.14, "Z", "o", "k", "per-epoch z-score (Z)"),
                                 (0.14, "G", "s", "white", "fixed per-channel scale (G)")):
        ba = [d["BA"][k][nm]["BA"] for k, _ in order]
        lo = [d["BA"][k][nm]["ci95"][0] for k, _ in order]
        hi = [d["BA"][k][nm]["ci95"][1] for k, _ in order]
        ax.errorbar(x + off, ba, yerr=[np.subtract(ba, lo), np.subtract(hi, ba)], fmt=mk, ms=3.5,
                    mfc=fc, mec="k", ecolor="k", elinewidth=0.7, capsize=1.5, label=lab)
    ax.axhline(1 / 3, color="0.4", lw=0.6)
    ax.axvspan(-0.5, 2.5, color="0.92", zorder=0)
    ax.set_ylim(0.25, 0.49)
    ax.text(1.0, 0.485, "before or at onset", ha="center", va="top", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], rotation=35, ha="right", fontsize=5.5)
    ax.set_title(title, fontsize=7, loc="left", pad=3)
    ax.grid(True, axis="y", color="0.93", lw=0.5); ax.tick_params(length=2, pad=1.5)


def main():
    d = json.load(open(AGG / "j6_tcrzem_prestimulus.json"))
    pl = AGG / "j6_tcrzem_prestimulus_long.json"
    fig, axes = plt.subplots(1, 2 if pl.exists() else 1, figsize=(7.16, 2.0), squeeze=False,
                             gridspec_kw={"width_ratios": [7, 9] if pl.exists() else [1]})
    panel(axes[0][0], d, ORDER, "(a) conference-matched epoch")
    if pl.exists():
        panel(axes[0][1], json.load(open(pl)), ORDER + LONG_EXTRA, "(b) long epoch (pre-specified)")
    axes[0][0].set_ylabel("clean BA", labelpad=1)
    axes[0][0].legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.tight_layout(pad=0.3, w_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
