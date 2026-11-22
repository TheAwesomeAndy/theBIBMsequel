"""Paper B, Fig. 1: the reference-size law on ERP CORE single trials (X12)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.style import apply  # noqa: E402

apply()
d = json.loads((ROOT / "outputs/X12/x12_results.json").read_text())
ms = d["provenance"]["ms"]; r = d["results"]; pars = list(r)
fig, ax = plt.subplots(1, len(pars), figsize=(7.16, 1.75), sharex=True)
for a, par in zip(ax, pars):
    obs = [r[par][f"center_ref_{m}"]["ba"] for m in ms]
    lo = [r[par][f"center_ref_{m}"]["ci"][0] for m in ms]; hi = [r[par][f"center_ref_{m}"]["ci"][1] for m in ms]
    pred = [r[par]["law"][str(m)]["predicted_ba"] for m in ms]
    a.fill_between(ms, lo, hi, color="#2a78d6", alpha=0.15, lw=0)
    a.plot(ms, obs, "o-", color="#2a78d6", ms=3, lw=1.1, label="observed")
    a.plot(ms, pred, ":", color="#111111", lw=1.2, label="law")
    a.axhline(r[par]["uncentered"]["ba"], color="#eb6834", lw=1.0, ls="--", label="uncentered")
    a.axhline(r[par]["center_all"]["ba"], color="0.45", lw=0.8, ls="-.", label="all trials")
    a.set_xscale("log", base=2); a.set_xticks(ms); a.set_xticklabels([str(m) for m in ms])
    a.set_title(par, loc="left"); a.set_xlabel("reference size $m$")
ax[0].set_ylabel("balanced accuracy"); ax[0].legend(frameon=False, loc="lower right", fontsize=6)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(ROOT / "paperB/figures/fig_law.pdf")
print("ok")
