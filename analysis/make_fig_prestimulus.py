#!/usr/bin/env python3
"""Journal figure: decoding from the empty pre-stimulus interval depends on normalization.

Reads outputs/aggregate/journal/j6_{shape,tcrzem}_prestimulus*.json: (a) SHAPE, (b) external
cohort at the conference-matched epoch, (c) external long epoch (adds 0..+2484 ms windows). Subject-level BA (95% CI) for
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
    top = max(0.49, max(d["BA"][k][nm]["ci95"][1] for k, _ in order for nm in ("Z", "G")) + 0.03)
    ax.set_ylim(0.25, top)
    ax.text(1.0, top - 0.005, "before or at onset", ha="center", va="top", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], rotation=35, ha="right", fontsize=5.5)
    ax.set_title(title, fontsize=7, loc="left", pad=3)
    ax.grid(True, axis="y", color="0.93", lw=0.5); ax.tick_params(length=2, pad=1.5)


def main():
    panels = [(f, o, t) for f, o, t in (
        ("j6_shape_prestimulus.json", ORDER, "(a) SHAPE"),
        ("j6_tcrzem_prestimulus.json", ORDER, "(b) external, matched epoch"),
        ("j6_tcrzem_prestimulus_long.json", ORDER + LONG_EXTRA, "(c) external, long epoch"))
        if (AGG / f).exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.1), squeeze=False,
                             gridspec_kw={"width_ratios": [len(o) for _, o, _ in panels]})
    for ax, (f, o, t) in zip(axes[0], panels):
        panel(ax, json.load(open(AGG / f)), o, t)
    axes[0][0].set_ylabel("clean BA", labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
