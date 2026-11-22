"""Fig. 3: ERP CORE time-resolved decoding (X02); Fig. 4: averaging law (X03)."""
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
LS = {"epoch_z": "-", "post_z": (0, (5, 1.5)), "pre_sd": "--", "none": ":"}


def fig3():
    r = json.loads((ROOT / "outputs/X02/x02_results.json").read_text())["results"]
    pars = list(r)
    fig, ax = plt.subplots(2, len(pars), figsize=(7.16, 2.4), sharex=True, sharey="row")
    ax = np.atleast_2d(ax)
    for j, par in enumerate(pars):
        t = np.array(r[par]["bin_center_ms"])
        for i, reg in enumerate(["across", "within"]):
            a = ax[i, j]
            for n in ["epoch_z", "post_z", "pre_sd", "none"]:
                lab, col, _ = SCALE[n]
                a.plot(t, r[par][reg][n]["time_course"], color=col, ls=LS[n], lw=1.1, label=lab)
            a.axvspan(-200, 0, color="0.93", lw=0, zorder=0)
            a.axvline(0, color="0.4", lw=0.7); a.axhline(0.5, color="0.3", lw=0.7, ls="-.")
            a.set_xlim(-200, 800)
            if i == 0:
                a.set_title(par, loc="left")
            if j == 0:
                a.set_ylabel(f"{reg}\nbalanced accuracy")
            if i == 1:
                a.set_xlabel("time (ms)")
    ax[0, 0].legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    fig.savefig(ROOT / "manuscript/figures/fig_erpcore.pdf"); print("wrote fig_erpcore.pdf")


def fig4():
    r = json.loads((ROOT / "outputs/X03/x03_results.json").read_text())["results"]
    pars = list(r)
    fig, ax = plt.subplots(1, len(pars), figsize=(7.16, 1.6), sharey=True)
    for j, par in enumerate(pars):
        a = ax[j]
        for n in ["epoch_z", "pre_sd"]:
            lab, col, _ = SCALE[n]
            ks = sorted(int(k) for k in r[par][n])
            m = [r[par][n][str(k)]["ba"] for k in ks]
            lo = [r[par][n][str(k)]["ci"][0] for k in ks]; hi = [r[par][n][str(k)]["ci"][1] for k in ks]
            a.fill_between(ks, lo, hi, color=col, alpha=0.18, lw=0)
            a.plot(ks, m, color=col, marker="o", ms=3, lw=1.2, ls=LS[n], label=f"observed, {lab}")
        kp = sorted(int(k) for k in r[par]["prediction_from_k1"])
        a.plot(kp, [r[par]["prediction_from_k1"][str(k)] for k in kp], color="0.15", lw=1.0, ls=(0, (1, 1)),
               label=r"prediction $\Phi(\sqrt{k}\,\Phi^{-1}(\mathrm{BA}_1))$")
        a.axhline(0.5, color="0.3", lw=0.7, ls="-.")
        a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8, 16]); a.set_xticklabels(["1", "2", "4", "8", "16"])
        a.set_title(par, loc="left"); a.set_xlabel("trials averaged, $k$")
        if j == 0:
            a.set_ylabel("pre-stimulus BA")
    ax[0].legend(frameon=False, loc="upper left", fontsize=5.8)
    fig.tight_layout(pad=0.3, w_pad=0.3)
    fig.savefig(ROOT / "manuscript/figures/fig_averaging.pdf"); print("wrote fig_averaging.pdf")


if __name__ == "__main__":
    if (ROOT / "outputs/X02/x02_results.json").exists():
        fig3()
    if (ROOT / "outputs/X03/x03_results.json").exists():
        fig4()
