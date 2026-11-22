"""Fig. 1: the leakage identity on SHAPE (aggregate quantities only).

(a) Grand-average ERP over channels and participants, per condition, in
    baseline-corrected microvolts.
(b) The same averages after per-epoch z-scoring: the pre-stimulus interval is
    offset by condition.
(c) Density of the pre-stimulus mean of each z-scored epoch and channel against
    -m/sd computed from the unscaled epoch (Proposition 1).

Usage: SHAPE_DATA_DIR=... python figures/make_fig_mechanism.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402
from figures.style import COND, apply  # noqa: E402

apply()
raw, y, g = cm.load_shape_raw()
X = cm.downsample(raw)
Z = cm.normalize(X, "epoch_z")
t = (np.arange(X.shape[1]) - cm.ONSET_DS) / cm.FS_DS * 1000

fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for k, (name, col, ls) in enumerate(COND):
    ax[0].plot(t, X[y == k].mean((0, 2)), color=col, ls=ls, lw=1.6, label=name)
    ax[1].plot(t, Z[y == k].mean((0, 2)), color=col, ls=ls, lw=1.6, label=name)
for a, lab in zip(ax[:2], [r"amplitude ($\mu$V)", "z-scored amplitude"]):
    a.axvspan(-200, 0, color="0.92", lw=0)
    a.axvline(0, color="0.4", lw=0.8)
    a.axhline(0, color="0.6", lw=0.6)
    a.set_xlabel("time (ms)"); a.set_ylabel(lab); a.set_xlim(-200, 800)
ax[0].set_title("(a) baseline-corrected", loc="left")
ax[1].set_title("(b) per-epoch z-scored", loc="left")
ax[0].legend(frameon=False, loc="upper left")

A = Z[:, 9:cm.ONSET_DS, :].mean(1).ravel()
B = (-X.mean(1) / X.std(1)).ravel()
r = np.corrcoef(A, B)[0, 1]
hb = ax[2].hexbin(B, A, gridsize=45, cmap="Greys", mincnt=1, bins="log", linewidths=0)
lim = [min(A.min(), B.min()), max(A.max(), B.max())]
ax[2].plot(lim, lim, color="#2a78d6", lw=1.0, ls="--")
ax[2].set_xlabel(r"$-m/\hat\sigma$ from the unscaled epoch")
ax[2].set_ylabel("pre-stimulus mean, z-scored")
ax[2].set_title(f"(c) identity, r = {r:.4f}", loc="left")
fig.tight_layout(pad=0.4)
out = ROOT / "manuscript" / "figures" / "fig_mechanism.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out, "r =", round(r, 5))
