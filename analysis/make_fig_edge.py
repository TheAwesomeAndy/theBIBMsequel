#!/usr/bin/env python3
"""Journal figure: the measured edge of stability across reservoir draws and thresholds.

Reads outputs/aggregate/journal/j4_shape_edge*.json (SHAPE: ten draws, order parameter and
accuracy; accuracy also with the windows measured from true onset) and
j4_tcrzem_edge_long.json (external cohort: ten draws for the order parameter, five for
accuracy on the long epoch).
(a) damage-spreading order parameter versus spectral radius: the ten SHAPE draws (thin),
    their mean (thick), and the mean of the ten external draws (dashed); shaded bands span
    the transitions rho* of each cohort's draws, dotted = the echo-state heuristic rho = 1;
(b) subject-level BA versus rho, mean over draws: SHAPE clean with the conference window
    (thin = single draws), with the onset window clean and with 30% of electrodes silent,
    with the post window, and the external long epoch; shaded = range of rho* (SHAPE);
    chance = 1/3;
(c) order parameter versus rho at three firing thresholds (SHAPE drives, draw 42), with
    each rho*.
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
JAG = ROOT / "outputs" / "aggregate" / "journal"
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_edge.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})


def load(name):
    p = JAG / name
    return json.load(open(p)) if p.exists() else None


def acc_matrix(res, cond):
    ra = res["rho_acc"]
    return ra, np.array([[res["draws"][d]["BA"][str(r)][cond]["BA"] for r in ra]
                         for d in res["draws"] if res["draws"][d].get("BA")])


def main():
    sh, ext = load("j4_shape_edge.json"), load("j4_tcrzem_edge_long.json")
    shp = load("j4_shape_edge_post.json")
    rf = sh["rho_fine"]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.8), gridspec_kw={"wspace": 0.38})

    Ds = np.array([[sh["draws"][d]["damage"][str(r)] for r in rf] for d in sh["draws"]])
    De = np.array([[ext["draws"][d]["damage"][str(r)] for r in rf] for d in ext["draws"]])
    for row in Ds:
        ax[0].plot(rf, row, color="0.7", lw=0.5)
    ax[0].plot(rf, Ds.mean(0), color="k", lw=1.4, label=f"SHAPE, {len(Ds)} draws")
    ax[0].plot(rf, De.mean(0), color="#009E73", lw=1.2, ls="--", label=f"external, {len(De)} draws")
    rs_s = [sh["draws"][d]["rho_star"] for d in sh["draws"]]
    rs_e = [ext["draws"][d]["rho_star"] for d in ext["draws"]]
    ax[0].axvspan(min(rs_s), max(rs_s), color="0.5", alpha=0.18, lw=0)
    ax[0].axvspan(min(rs_e), max(rs_e), facecolor="none", edgecolor="#009E73", hatch="////", lw=0, alpha=0.5)
    ax[0].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[0].set_ylabel("damage (Hamming)", labelpad=1)
    ax[0].set_title(f"(a) order parameter, $\\rho^*$={np.mean(rs_s):.2f} / {np.mean(rs_e):.2f}",
                    fontsize=7, loc="left", pad=3)
    ax[0].legend(frameon=False, fontsize=5.3, loc="lower right")

    sho = load("j4_shape_edge_onset.json")
    ra, Mc = acc_matrix(sh, "clean")
    for row in Mc:
        ax[1].plot(ra, row, color="0.75", lw=0.4)
    ax[1].plot(ra, Mc.mean(0), color="k", lw=1.4, marker="o", ms=2.5, label="SHAPE, conference window")
    if sho is not None and sho.get("summary"):
        _, Moc = acc_matrix(sho, "clean")
        _, Mos = acc_matrix(sho, "signal30")
        ax[1].plot(ra, Moc.mean(0), color="#0072B2", lw=1.2, marker="s", ms=2.5, label="SHAPE, onset window")
        ax[1].plot(ra, Mos.mean(0), color="#0072B2", lw=1.0, ls="--", marker="s", ms=2.5, mfc="white",
                   label="onset window, 30% silent")
    if shp is not None and shp.get("summary"):
        _, Mp = acc_matrix(shp, "clean")
        ax[1].plot(ra, Mp.mean(0), color="#CC79A7", lw=1.1, ls="-.", marker="D", ms=2.3,
                   label="SHAPE, post window")
    _, Me = acc_matrix(ext, "clean")
    ax[1].plot(ext["rho_acc"], Me.mean(0), color="#009E73", lw=1.1, ls=":", marker="^", ms=2.5,
               label="external, long epoch")
    ax[1].axvspan(min(rs_s), max(rs_s), color="0.5", alpha=0.18, lw=0)
    ax[1].axhline(1 / 3, color="0.4", lw=0.6)
    ax[1].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[1].set_ylabel("BA (mean over draws)", labelpad=1)
    ax[1].set_title("(b) accuracy versus $\\rho$", fontsize=7, loc="left", pad=3)
    ax[1].legend(frameon=False, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
                 handlelength=2.2, columnspacing=0.8)

    em = sh.get("edge_maps_draw42", {}).get("theta", {})
    for (th, ent), ls in zip(sorted(em.items(), key=lambda kv: float(kv[0])), ("-", "--", ":")):
        dm = [ent["damage"][str(r)] for r in rf]
        rsx = ent["rho_star"]
        ax[2].plot(rf, dm, color="k", lw=1.0, ls=ls, label=f"$\\vartheta$={th}: $\\rho^*$={rsx:.2f}")
        ax[2].plot([rsx], [0.5 * max(dm)], marker="o", ms=3, color="k")
    ax[2].legend(frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
                 handlelength=2.2, columnspacing=0.8)
    ax[2].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[2].set_xlabel("spectral radius $\\rho$", labelpad=1); ax[2].set_ylabel("damage (Hamming)", labelpad=1)
    ax[2].set_title("(c) edge versus threshold (SHAPE)", fontsize=7, loc="left", pad=3)
    for a in ax:
        a.grid(True, color="0.93", lw=0.5); a.tick_params(length=2, pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}")


if __name__ == "__main__":
    main()
