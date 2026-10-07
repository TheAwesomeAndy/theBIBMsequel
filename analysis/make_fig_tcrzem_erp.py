#!/usr/bin/env python3
"""Journal figure: grand-average affective ERPs of the external TCRZEM cohort (228 subjects).

Uses the pre-specified long epoch (-200..+2500 ms, baseline-corrected microvolts, no
z-scoring) so the figure shows where the dataset's effects lie relative to the
conference-matched analysis window, which ends at +800 ms.
(a) grand-average waveforms per valence at the channel with the largest
    emotional-minus-neutral difference over 500-1300 ms (the published salience window);
    the conference-matched window end (+800 ms) is marked;
(b) emotional-minus-neutral difference over all 32 channels;
(c) negative-minus-pleasant difference over all 32 channels (published valence window
    1500-2500 ms marked).
Group averages only (no subject-level data). Requires $TCRZEM_EPOCH=long data (the cache
is built on first use).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TCRZEM_EPOCH"] = "long"
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcrzem_data as TD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("FIG_OUT", ROOT / "manuscript/figures/journal/fig_tcrzem_erp.pdf"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
NAMES = {0: "negative", 1: "neutral", 2: "pleasant"}
STYLE = {0: ("#0072B2", "-"), 1: ("#7f7f7f", ":"), 2: ("#D55E00", "--")}


def diffmap(ax, D, t, names, title, marks, signs):
    vmax = np.percentile(np.abs(D), 98)
    im = ax.imshow(D.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   norm=colors.TwoSlopeNorm(0, -vmax, vmax), extent=[t[0], t[-1], 0, D.shape[1]])
    for m in marks:
        ax.axvline(m, color="k", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    for tx, lab in signs:
        ax.text(tx, 3.5, lab, ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.5))
    ax.set_xlabel("time (ms)", labelpad=1); ax.set_ylabel("channel", labelpad=1)
    ax.set_title(title, fontsize=7, loc="left", pad=3)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.set_label("$\\mu$V", fontsize=5.5, labelpad=1)


def main():
    X, y, g = TD.load_cohort(zscore=False)
    names, _ = TD.montage()
    t = (np.arange(X.shape[1]) - TD.onset_sample()) / 256 * 1000
    ga = {c: X[y == c].mean(0) for c in (0, 1, 2)}
    emo = (ga[0] + ga[2]) / 2
    sal = (t >= 500) & (t <= 1300)
    ch = int(np.argmax(np.abs((emo - ga[1])[sal].mean(0))))

    fig = plt.figure(figsize=(7.16, 1.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1], wspace=0.42)
    ax0 = fig.add_subplot(gs[0])
    for c in (0, 1, 2):
        col, ls = STYLE[c]
        ax0.plot(t, ga[c][:, ch], color=col, ls=ls, lw=1.0, label=NAMES[c])
    ax0.axvspan(500, 1300, color="0.92", zorder=0); ax0.axvspan(1500, 2500, color="0.85", zorder=0)
    ax0.axvline(800, color="k", lw=0.7, ls="-."); ax0.axvline(0, color="k", lw=0.6, ls="--")
    ax0.axhline(0, color="0.6", lw=0.6)
    ax0.text(810, ax0.get_ylim()[1] * 0.92, "+800 ms", fontsize=5.5, va="top")
    ax0.set_xlabel("time (ms)", labelpad=1); ax0.set_ylabel("amplitude ($\\mu$V)", labelpad=1)
    ax0.set_title(f"(a) grand average, {names[ch]}", fontsize=7, loc="left", pad=3)
    ax0.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=2.0, borderaxespad=0.2)
    ax0.grid(True, color="0.93", lw=0.5)
    diffmap(fig.add_subplot(gs[1]), emo - ga[1], t, names, "(b) emotional $-$ neutral",
            (500, 1300, 800), ((900, "$+$"),))
    neg_pos = ga[0] - ga[2]
    late = (t >= 1500) & (t <= 2500)
    sgn = "$+$" if neg_pos[late].mean() > 0 else "$-$"
    diffmap(fig.add_subplot(gs[2]), neg_pos, t, names, "(c) negative $-$ pleasant",
            (1500, 2500, 800), ((2000, sgn),))
    for a in fig.axes[:1]:
        a.tick_params(length=2, pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    # aggregate effect-size summary (no subject-level values are written)
    import json
    Xs = {c: X[y == c][:, :, ch] for c in (0, 1, 2)}   # rows in subject order (tcrzem_data.average)
    sal_amp = {c: Xs[c][:, sal].mean(1) for c in (0, 1, 2)}
    late_m = (t >= 1500) & (t <= 2500)
    d_sal = (sal_amp[0] + sal_amp[2]) / 2 - sal_amp[1]
    d_val = Xs[0][:, late_m].mean(1) - Xs[2][:, late_m].mean(1)
    summ = {"channel": names[ch],
            "salience_500_1300ms_uV": float(d_sal.mean()),
            "salience_within_subject_dz": float(d_sal.mean() / d_sal.std(ddof=1)),
            "neutral_amplitude_between_subject_sd_uV": float(sal_amp[1].std(ddof=1)),
            "valence_1500_2500ms_uV": float(d_val.mean()),
            "valence_within_subject_dz": float(d_val.mean() / d_val.std(ddof=1)),
            "n_subjects": int(len(d_sal))}
    jp = ROOT / "outputs" / "aggregate" / "journal" / "tcrzem_erp_summary.json"
    json.dump(summ, open(jp, "w"), indent=2)
    print(summ)
    print(f"[fig] wrote {OUT}; salience channel = {names[ch]}; "
          f"emotional-neutral 500-1300 ms = {(emo - ga[1])[sal, ch].mean():.2f} uV; "
          f"negative-pleasant 1500-2500 ms = {neg_pos[late, ch].mean():.2f} uV")


if __name__ == "__main__":
    main()
