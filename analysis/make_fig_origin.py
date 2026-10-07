#!/usr/bin/env python3
"""Journal figure: the dropout score is a function of where the fill sits (Proposition 1).

Reads outputs/aggregate/journal/j5_tcrzem_origin*.json. For each fixed encoder, subject-level
BA (95% CI) at 30% channel dropout versus the standardized position kappa of the fill
(kappa = 0 is train-mean fill), with the clean BA (dashed, identical for every origin by
construction) and the encoder's native zero (marker at its mean standardized position
-|mu|/sigma of the dropped block). Panels: one per available epoch (conference-matched,
pre-specified long epoch). Encoders differ by marker and line style (grayscale-legible).
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
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_origin.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
STY = {"Band-power": ("#E69F00", "s", "-"), "ERP-window": ("#0072B2", "o", "--"),
       "Reservoir": ("#009E73", "^", ":")}
PANELS = [("j5_tcrzem_origin.json", "(a) conference-matched epoch"),
          ("j5_tcrzem_origin_long.json", "(b) long epoch (pre-specified)")]


def main():
    panels = [(f, t) for f, t in PANELS if (AGG / f).exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels) / 1.0 if len(panels) > 1 else 3.5, 1.9),
                             squeeze=False)
    for ax, (f, title) in zip(axes[0], panels):
        d = json.load(open(AGG / f))
        ks = d["kappas"]
        for n, (col, mk, ls) in STY.items():
            e = d["dropout"][n]["30"]
            ba = [e[f"kappa_{k:+.1f}"]["BA"] for k in ks]
            lo = [e[f"kappa_{k:+.1f}"]["ci95"][0] for k in ks]
            hi = [e[f"kappa_{k:+.1f}"]["ci95"][1] for k in ks]
            ax.plot(ks, ba, color=col, marker=mk, ls=ls, lw=1.0, ms=3, label=n)
            ax.fill_between(ks, lo, hi, color=col, alpha=0.12, lw=0)
            zpos = -d["predicted_logit_shift_30"][n]["mean_abs_mu_over_sigma_dropped"]
            ax.plot([zpos], [e["native_zero"]["BA"]], marker=mk, ms=6, mfc="white", mec=col, mew=1.2, ls="none")
            ax.axhline(d["clean_BA"][n]["BA"], color=col, lw=0.6, ls=(0, (1, 2)))
        ax.axhline(1 / 3, color="0.4", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_xlabel("standardized fill position $\\kappa$ (0 = train mean)", labelpad=1)
        ax.set_ylabel("BA at 30% dropout", labelpad=1)
        ax.set_title(title, fontsize=7, loc="left", pad=3)
        ax.grid(True, color="0.93", lw=0.5)
        ax.tick_params(length=2, pad=1.5)
    axes[0][0].legend(frameon=False, fontsize=5.5, loc="lower center", ncol=3, handlelength=2.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
