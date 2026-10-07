#!/usr/bin/env python3
"""Journal figure: the fill rule alone, on both cohorts (feature-coordinate experiment).

Paired subject-level change in BA (95% CI) of train-mean fill minus zero-fill against the
fraction of electrodes dropped, for the three fixed encoders. (a) SHAPE (conference E2/E5
outputs); (b) external cohort, pre-specified long epoch (j1_tcrzem_core_long.json);
(c) external cohort, conference-matched epoch (j1_tcrzem_core.json). The legend of each
panel gives the measured mean |mu|/sigma of each encoder on that cohort, the quantity that
sets the gap by Proposition 1. Encoders differ by marker and line style (grayscale-legible).
Reads only aggregate JSON.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "aggregate"
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_fills.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
ENC = ["ERP-window", "Band-power", "Reservoir"]
STYLE = {"ERP-window": dict(color="#D55E00", marker="o", ls="-"),
         "Band-power": dict(color="#0072B2", marker="s", ls="--"),
         "Reservoir": dict(color="#009E73", marker="^", ls=":")}
DODGE = {"ERP-window": -1.4, "Band-power": 0.0, "Reservoir": 1.4}
LEVELS = [10, 20, 30, 40, 50]


def panels():
    e2 = json.load(open(A / "e2_trainonly_pca_fills.json"))
    e5 = json.load(open(A / "e5_controls.json"))
    out = [("(a) SHAPE", e2["fill_minus_zero"], e5["centeredness_mean_abs_mu_over_sigma"])]
    for f, lab in (("j1_tcrzem_core_long.json", "(b) external, long epoch"),
                   ("j1_tcrzem_core.json", "(c) external, $-$200..800 ms")):
        p = A / "journal" / f
        if p.exists():
            d = json.load(open(p))
            out.append((lab, d["fill_minus_zero"], d["centeredness_mean_abs_mu_over_sigma"]))
    return out


def main():
    P = panels()
    fig, axes = plt.subplots(1, len(P), figsize=(7.16, 1.85), sharey=True)
    for ax, (title, fmz, cent) in zip(axes, P):
        ax.axhline(0, color="k", lw=0.6)
        for enc in ENC:
            r = fmz[enc]["mean"]
            m = [r[str(l)]["mean_diff"] for l in LEVELS]
            lo = [r[str(l)]["mean_diff"] - r[str(l)]["ci95"][0] for l in LEVELS]
            hi = [r[str(l)]["ci95"][1] - r[str(l)]["mean_diff"] for l in LEVELS]
            st = STYLE[enc]
            ax.errorbar([l + DODGE[enc] for l in LEVELS], m, yerr=[lo, hi], color=st["color"],
                        marker=st["marker"], ls=st["ls"], lw=0.9, ms=3.0, capsize=1.5, elinewidth=0.6,
                        mfc="white" if enc == "Reservoir" else st["color"],
                        label=f"{enc} ($|\\mu|/\\sigma$ {cent[enc]['mean']:.2f})")
        ax.set_title(title, fontsize=7, loc="left", pad=3)
        ax.set_xticks(LEVELS); ax.set_xlim(5, 55)
        ax.set_xlabel("electrodes dropped (%)", labelpad=1)
        ax.set_ylim(-0.05, 0.125)
        ax.legend(frameon=False, fontsize=5.2, loc="upper left", handlelength=2.0, ncol=1, borderaxespad=0.2)
        ax.grid(True, color="0.93", lw=0.5); ax.tick_params(length=2, pad=1.5)
    axes[0].set_ylabel("$\\Delta$BA, mean $-$ zero fill", labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
