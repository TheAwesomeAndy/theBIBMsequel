"""Fig. 2: SHAPE results from outputs/X01, X04, X05 (no restricted data needed).

(a) Time-resolved three-class decoding (X05) under three scalings, 95% CIs.
(b) Balanced accuracy of each encoder under three scalings (X01, X04), 95% CIs.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.style import SCALE, apply  # noqa: E402

apply()
x5 = json.loads((ROOT / "outputs/X05/x05_results.json").read_text())
x1 = json.loads((ROOT / "outputs/X01/x01_results.json").read_text())["balanced_accuracy"]
p4 = ROOT / "outputs/X04/x04_results.json"
x4 = json.loads(p4.read_text())["balanced_accuracy"] if p4.exists() else {}

fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.9), gridspec_kw={"width_ratios": [1, 1.35]})
t = np.array(x5["bin_center_ms"])
for n in ["epoch_z", "pre_sd", "none"]:
    lab, col, _ = SCALE[n]
    ls = "-" if n == "epoch_z" else ("--" if n == "pre_sd" else ":")
    ax[0].fill_between(t, x5[n]["ci_lo"], x5[n]["ci_hi"], color=col, alpha=0.18, lw=0)
    ax[0].plot(t, x5[n]["ba"], color=col, lw=1.4, ls=ls, label=lab)
ax[0].axvspan(-200, 0, color="0.93", lw=0, zorder=0)
ax[0].axvline(0, color="0.4", lw=0.8); ax[0].axhline(1 / 3, color="0.3", lw=0.8, ls="-.")
ax[0].set_xlim(-200, 800); ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("balanced accuracy")
ax[0].set_title("(a) time-resolved decoding", loc="left"); ax[0].legend(frameon=False, loc="upper left")

enc = [("pre_mean", "pre-stim.\nmean", x1), ("reservoir_pre", "reservoir\nto onset", x1),
       ("reservoir_bibm", "reservoir\n(BIBM)", x1), ("erp_windows", "ERP\nwindows", x1)]
if x4:
    enc += [("eegnet_pre", "EEGNet\npre-stim.", x4), ("eegnet_full", "EEGNet\nfull", x4)]
norms = ["epoch_z", "pre_sd", "none"] if not x4 else ["epoch_z", "pre_sd"]
w = 0.8 / len(norms)
for i, n in enumerate(norms):
    lab, col, hatch = SCALE[n]
    xs, ys, lo, hi = [], [], [], []
    for j, (key, _, src) in enumerate(enc):
        k = f"{key}|{n}"
        if k not in src:
            continue
        xs.append(j + (i - (len(norms) - 1) / 2) * w); ys.append(src[k]["ba"])
        lo.append(src[k]["ba"] - src[k]["ci"][0]); hi.append(src[k]["ci"][1] - src[k]["ba"])
    ax[1].bar(xs, ys, w * 0.92, color=col, hatch=hatch, edgecolor="white", lw=0, label=lab)
    ax[1].errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="0.2", lw=0.7, capsize=1.5)
ax[1].axhline(1 / 3, color="0.3", lw=0.8, ls="-.")
ax[1].set_xticks(range(len(enc))); ax[1].set_xticklabels([e[1] for e in enc])
ax[1].set_ylim(0.2, 0.8); ax[1].set_ylabel("balanced accuracy")
ax[1].set_title("(b) encoders under each scaling", loc="left"); ax[1].legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout(pad=0.4)
out = ROOT / "manuscript" / "figures" / "fig_shape.pdf"
fig.savefig(out); print("wrote", out)
