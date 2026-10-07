#!/usr/bin/env python3
"""Journal figure: the measured edge of stability across reservoir draws and thresholds.

Reads outputs/aggregate/journal/j4_tcrzem_edge*.json (external cohort) and the conference
outputs/aggregate/e1_rho_sweep.json (SHAPE, draw 42).
(a) damage-spreading order parameter versus spectral radius for ten reservoir draws on the
    external drives (thin), their mean (thick), and the SHAPE curve (dashed); the mean
    transition rho* (shaded band = range over draws) and the echo-state heuristic rho = 1
    (dotted) are marked;
(b) subject-level clean BA versus rho for the draws with an accuracy sweep (thin) and their
    mean (thick) on the epoch(s) available, with SHAPE (draw 42, 95% CI); chance = 1/3;
(c) order parameter versus rho at three firing thresholds (draw 42) with each rho*.
Line styles and markers carry the distinctions (grayscale-legible).
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
AGG = ROOT / "outputs" / "aggregate"
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_edge.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    # J4 ran with the long epoch: the order parameter uses the conference-matched drives and
    # the conference definition; the accuracy sweep uses the pre-specified long epoch.
    std = json.load(open(AGG / "journal" / "j4_tcrzem_edge_long.json"))
    lng = None
    e1 = json.load(open(AGG / "e1_rho_sweep.json"))
    rf = std["rho_fine"]; draws = list(std["draws"])
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.75), gridspec_kw={"wspace": 0.38})

    D = np.array([[std["draws"][d]["damage"][str(r)] for r in rf] for d in draws])
    for row in D:
        ax[0].plot(rf, row, color="0.75", lw=0.5)
    ax[0].plot(rf, D.mean(0), color="#009E73", lw=1.4, label=f"external, {len(draws)} draws")
    ax[0].plot(rf, [e1["damage"][str(r)] for r in rf], color="k", lw=1.0, ls="--", label="SHAPE, draw 42")
    rs = [std["draws"][d]["rho_star"] for d in draws]
    ax[0].axvspan(min(rs), max(rs), color="#009E73", alpha=0.15, lw=0)
    ax[0].axvline(np.mean(rs), color="#009E73", lw=0.8)
    ax[0].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[0].set_ylabel("damage (Hamming)", labelpad=1)
    ax[0].set_title(f"(a) order parameter, $\\rho^*$={np.mean(rs):.2f}", fontsize=7, loc="left", pad=3)
    ax[0].legend(frameon=False, fontsize=5.5, loc="lower right")

    def sweep(a, res, col, ls, lab):
        ra = res["rho_acc"]
        M = np.array([[res["draws"][d]["BA"][str(r)]["clean"]["BA"] for r in ra]
                      for d in res["draws"] if res["draws"][d].get("BA")])
        for row in M:
            a.plot(ra, row, color=col, lw=0.4, alpha=0.5, ls=ls)
        a.plot(ra, M.mean(0), color=col, lw=1.4, ls=ls, marker="o", ms=2.5, label=lab)
    sweep(ax[1], std, "#009E73", "-", "external, long epoch")
    if lng is not None:
        sweep(ax[1], lng, "#0072B2", "-.", "external, long epoch")
    rg = e1["rho_grid"]
    y = [e1["BA"][str(r)]["clean"]["BA"] for r in rg]
    lo = [e1["BA"][str(r)]["clean"]["ci95"][0] for r in rg]; hi = [e1["BA"][str(r)]["clean"]["ci95"][1] for r in rg]
    ax[1].errorbar(rg, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color="k", ls="--", lw=0.9,
                   marker="s", ms=2.5, capsize=1.5, elinewidth=0.6, label="SHAPE, draw 42")
    ax[1].axhline(1 / 3, color="0.4", lw=0.6)
    ax[1].axvline(np.mean(rs), color="#009E73", lw=0.8); ax[1].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[1].set_ylabel("clean BA", labelpad=1)
    ax[1].set_title("(b) accuracy versus $\\rho$", fontsize=7, loc="left", pad=3)
    ax[1].legend(frameon=False, fontsize=5, loc="upper right")

    em = std.get("edge_maps_draw42", {}).get("theta", {})
    for (th, ent), ls in zip(sorted(em.items(), key=lambda kv: float(kv[0])), ("-", "--", ":")):
        dm = [ent["damage"][str(r)] for r in rf]
        ax[2].plot(rf, dm, color="k", lw=1.0, ls=ls, label=f"$\\vartheta$={th}: $\\rho^*$={ent['rho_star']:.2f}")
        ax[2].plot([ent["rho_star"]], [0.5 * max(dm)], marker="o", ms=3, color="k")
    ax[2].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[2].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[2].set_ylabel("damage (Hamming)", labelpad=1)
    ax[2].set_title("(c) edge versus threshold", fontsize=7, loc="left", pad=3)
    ax[2].legend(frameon=False, fontsize=5.5, loc="lower right")
    for a in ax:
        a.grid(True, color="0.93", lw=0.5); a.tick_params(length=2, pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
