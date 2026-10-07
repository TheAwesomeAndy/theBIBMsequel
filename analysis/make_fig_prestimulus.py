#!/usr/bin/env python3
"""Journal figure: decoding from the empty pre-stimulus interval depends on normalization.

Reads outputs/aggregate/journal/j6_tcrzem_prestimulus.json. Subject-level BA (95% CI) for
every encoder window under per-epoch z-scoring (Z, conference convention; filled markers)
and train-only per-channel scaling (G; open markers). Windows before onset carry no
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


def main():
    d = json.load(open(AGG / "j6_tcrzem_prestimulus.json"))
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(len(ORDER))
    for off, nm, mk, fc, lab in ((-0.14, "Z", "o", "k", "per-epoch z-score (Z)"),
                                 (0.14, "G", "s", "white", "train-only scale (G)")):
        ba = [d["BA"][k][nm]["BA"] for k, _ in ORDER]
        lo = [d["BA"][k][nm]["ci95"][0] for k, _ in ORDER]
        hi = [d["BA"][k][nm]["ci95"][1] for k, _ in ORDER]
        ax.errorbar(x + off, ba, yerr=[np.subtract(ba, lo), np.subtract(hi, ba)], fmt=mk, ms=3.5,
                    mfc=fc, mec="k", ecolor="k", elinewidth=0.7, capsize=1.5, label=lab)
    ax.axhline(1 / 3, color="0.4", lw=0.6)
    ax.axvspan(-0.5, 2.5, color="0.92", zorder=0)
    ax.text(1.0, ax.get_ylim()[1], "window before or at onset", ha="center", va="top", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ORDER], rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("clean BA", labelpad=1)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax.grid(True, axis="y", color="0.93", lw=0.5); ax.tick_params(length=2, pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
