#!/usr/bin/env python3
"""Fig. 1 (camera-ready): the fixed LIF reservoir as a dynamical system, linked to accuracy.

(a) Spike raster for one condition's grand-average drive (shaded = analysis window).
(b) Population state-space trajectory (first three principal components of the membrane
    state) for the three affective conditions, colored by time.
(c) Subject-level balanced accuracy versus spectral radius rho (E1): clean and 30%
    signal-level electrode removal, with 95% subject-bootstrap CIs; rho = 0 removes the
    recurrence.
(d) Damage-spreading order parameter versus rho (E1): mean normalized Hamming distance
    after a one-spike flip; the measured transition and the operating point rho = 0.9 are
    marked, with the echo-state heuristic rho = 1 for reference.

Replaces the earlier eigenvalue-spectrum and state-separation panels, which reviewers noted
were properties of the construction rather than results. Reads the restricted SHAPE pickle
locally only to draw (a)-(b) from grand averages; (c)-(d) read aggregate JSON.
Grayscale-legible: series differ by marker and line style, not color alone.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_camera_ready as EC  # noqa: E402
import reanalysis_subject_bootstrap as RB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/imported/fig_reservoir_dyn.pdf"))
E1 = json.load(open(ROOT / "outputs/aggregate/e1_rho_sweep.json"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "cm"})


def trace(x, rho=0.9):
    W_in, W = EC.weights(rho); T = len(x)
    mem = np.zeros(EC.N_RES); spk = np.zeros(EC.N_RES)
    S = np.zeros((T, EC.N_RES)); M = np.zeros((T, EC.N_RES))
    for t in range(T):
        mem = (1.0 - EC.BETA) * mem * (1.0 - spk) + W_in[:, 0] * x[t] + W @ spk
        spk = (mem >= EC.THETA).astype(float)
        mem = np.maximum(mem - spk * EC.THETA, 0.0)
        S[t] = spk; M[t] = mem
    return S, M


def transition(dmg):
    """rho at which the damage, linearly interpolated between grid points, reaches half its maximum."""
    keys = sorted(dmg, key=float)
    r = np.array([float(k) for k in keys]); v = np.array([dmg[k] for k in keys])
    i = int(np.argmax(v >= 0.5 * v.max()))
    if i == 0:
        return float(r[0]), r, v
    rstar = r[i - 1] + (0.5 * v.max() - v[i - 1]) / (v[i] - v[i - 1]) * (r[i] - r[i - 1])
    return float(rstar), r, v


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X = np.asarray(d["X_ds"], float); y = np.asarray(d["y"])
    t_ms = np.arange(X.shape[1]) / RB.FS * 1000
    ga = {c: X[y == c].mean(0) for c in (0, 1, 2)}
    lpp = (t_ms >= 450) & (t_ms <= 800)
    ch = int(np.argmax(np.abs(((ga[0] + ga[2]) / 2 - ga[1])[lpp].mean(0))))
    tr = {c: trace(ga[c][:, ch]) for c in (0, 1, 2)}
    pca = PCA(3).fit(np.vstack([tr[c][1] for c in (0, 1, 2)]))
    names = {0: "neg", 1: "neu", 2: "pos"}

    fig = plt.figure(figsize=(7.16, 1.85))                 # one row across both IEEE columns
    gs = fig.add_gridspec(1, 4, wspace=0.50, width_ratios=[1.0, 1.0, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    st, sn = np.where(tr[0][0][:, :110] > 0)
    ax.scatter(st, sn, s=0.8, marker="|", color="black", linewidths=0.35)
    ax.axvspan(EC.T_START, EC.T_END, color="0.85", zorder=0)
    ax.set_xlim(0, X.shape[1]); ax.set_ylim(0, 110)
    ax.set_xlabel("time step", labelpad=1); ax.set_ylabel("reservoir unit", labelpad=1)
    ax.set_title("(a) LIF spike raster", fontsize=7, loc="left", pad=3)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ax.set_box_aspect(None, zoom=1.0)   # color runs from epoch start (dark) to the end of the epoch (light)
    tn = np.linspace(0, 1, X.shape[1])
    for c, mk in zip((0, 1, 2), ("o", "s", "^")):
        P = pca.transform(tr[c][1]); pts = P.reshape(-1, 1, 3)
        lc = Line3DCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="viridis", lw=0.8)
        lc.set_array(tn[:-1]); ax.add_collection3d(lc)
        ax.scatter(*P[0], color="white", edgecolor="k", s=12, marker=mk, label=names[c], depthshade=False,
                   linewidths=0.6)
    allP = np.vstack([pca.transform(tr[c][1]) for c in (0, 1, 2)])
    ax.set_xlim(allP[:, 0].min(), allP[:, 0].max()); ax.set_ylim(allP[:, 1].min(), allP[:, 1].max())
    ax.set_zlim(allP[:, 2].min(), allP[:, 2].max()); ax.view_init(elev=22, azim=-58)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])   # PC scales carry no meaning
    ax.set_xlabel("PC1", labelpad=-14, fontsize=5.5); ax.set_ylabel("PC2", labelpad=-14, fontsize=5.5)
    ax.set_zlabel("PC3", labelpad=-17, fontsize=5.5)
    ax.legend(fontsize=5, loc="upper left", title="start", title_fontsize=5, frameon=False,
              borderaxespad=0.0, handletextpad=0.2, bbox_to_anchor=(-0.08, 1.02))
    ax.set_title("(b) state-space trajectory", fontsize=7, loc="left", pad=3)

    rhos = [float(r) for r in E1["rho_grid"]]
    ax = fig.add_subplot(gs[0, 2])
    for key, lab, mk, ls in (("clean", "clean", "o", "-"), ("signal", "30% removed", "^", "--")):
        b = np.array([E1["BA"][str(r)][key]["BA"] for r in rhos])
        lo = np.array([E1["BA"][str(r)][key]["ci95"][0] for r in rhos])
        hi = np.array([E1["BA"][str(r)][key]["ci95"][1] for r in rhos])
        ax.errorbar(rhos, b, yerr=[b - lo, hi - b], fmt=mk + ls, color="black", ms=2.8, lw=0.8,
                    capsize=1.5, elinewidth=0.6, mfc="white" if ls == "--" else "black", label=lab)
    ax.axhline(1 / 3, color="0.4", lw=0.6, ls=":"); ax.text(rhos[0], 1 / 3 + 0.004, "chance", ha="left", va="bottom", fontsize=5.5)
    ax.axvline(0.9, color="0.3", lw=0.7, ls="-.")
    ax.set_xlabel(r"spectral radius $\rho$", labelpad=1); ax.set_ylabel("balanced accuracy", labelpad=1)
    ax.set_title(r"(c) accuracy versus $\rho$", fontsize=7, loc="left", pad=3)
    ax.set_ylim(0.31, 0.60)
    ax.legend(fontsize=5.2, loc="upper right", frameon=False, handlelength=2.0, borderaxespad=0.1)

    ax = fig.add_subplot(gs[0, 3])
    rstar, r, v = transition(E1["damage"])
    ax.axvspan(r.min(), rstar, color="0.93", zorder=0); ax.axvspan(rstar, r.max(), color="0.82", zorder=0)
    ax.plot(r, v, "-o", color="black", ms=2.2, lw=0.8)
    ax.axvline(1.0, color="0.4", lw=0.7, ls=":"); ax.axvline(0.9, color="0.3", lw=0.7, ls="-.")
    ax.text(0.04, 0.90, "ordered", transform=ax.transAxes, fontsize=5.5)
    ax.text(0.96, 0.06, "irregular", transform=ax.transAxes, fontsize=5.5, ha="right")
    ax.set_xlabel(r"spectral radius $\rho$", labelpad=1); ax.set_ylabel("damage after one-spike flip", labelpad=1)
    ax.set_title(r"(d) transition, $\rho^\ast\!\approx\!%.2f$" % rstar, fontsize=7, loc="left", pad=3)
    ax.set_xlim(r.min(), r.max())

    for a in fig.axes:
        if a.name != "3d":
            a.tick_params(length=2, pad=1.5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] wrote {OUT}; affect channel={ch}; rho*={rstar:.2f}")


if __name__ == "__main__":
    main()
