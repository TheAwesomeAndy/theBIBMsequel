#!/usr/bin/env python3
"""J9: does the reservoir read the per-epoch normalization offset? (pre-specified, memory #44)

Under per-epoch z-scoring the pre-stimulus mean of every channel is an exact linear
function of its post-stimulus mean (Proposition 2), so a reservoir window that sits
mostly before onset can carry post-stimulus amplitude through that offset. Reservoir
draw 42 at rho = 0.9 on the conference-matched epoch (onset at sample 51):

(a) pre-specified: Pearson r, per channel across observations, between the total spike
    count in the conference window [10, 70) and the post-stimulus mean m (samples
    51..255) of the z-scored epoch; median and range over channels;
(b) the same against |m| (the membrane is rectified and W_in symmetric, so a signed r
    can miss a V-shaped dependence);
(c) out-of-fold R^2 of m from the channel's BSC6 code (ridge, alpha = 1, GroupKFold(5)
    by subject, all channels pooled), for a strictly pre-stimulus window [3, 51) and the
    conference window, under Z (per-epoch z-score, the conference input) and G (the
    microvolt epoch divided by its per-channel SD pooled over all observations, label-free).
    R^2 > 0 for the pre-stimulus window under Z but not under G means the reservoir code
    carries the post-stimulus amplitude through the normalization alone.
Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402

ONSET = 51
WINDOWS = {"pre": (3, 51), "conference": (10, 70)}


def oof_r2(F, m, groups):
    """Out-of-fold R^2 of m (n,) from features F (n, d), GroupKFold(5) by subject."""
    pred = np.zeros_like(m)
    for tr, te in GroupKFold(5).split(F, m, groups):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        reg = Ridge(alpha=1.0).fit((F[tr] - mu) / sd, m[tr])
        pred[te] = reg.predict((F[te] - mu) / sd)
    return float(1.0 - np.sum((m - pred) ** 2) / np.sum((m - m.mean()) ** 2))


def main():
    t0 = time.time()
    Xuv, y, g = J.load_cohort(zscore=False)
    Xuv = Xuv[:, :256, :]
    Z = J.zscore_epochs(Xuv)
    G = Xuv / Xuv.std(axis=(0, 1))
    N, T, n_ch = Z.shape
    m = Z[:, ONSET:, :].mean(axis=1)                       # (N, ch) post-stimulus mean (z units)
    pre = Z[:, :ONSET, :].mean(axis=1)
    res = {"dataset": J.DATASET, "draw": 42, "rho": 0.9, "windows": WINDOWS,
           "identity_max_abs(51*pre+205*post)": float(np.abs(ONSET * pre + (T - ONSET) * m).max()),
           "protocol": __doc__.split("\n\n")[1].strip()}
    groups = np.repeat(g, n_ch)
    for inp, X in (("Z", Z), ("G", G)):
        for wn, (a, b) in WINDOWS.items():
            B, rate = J.reservoir_codes(X, t_start=a, t_end=b)
            cnt = B.sum(axis=2)                              # (N, ch) total spikes in window
            r_signed = [float(np.corrcoef(cnt[:, c], m[:, c])[0, 1]) for c in range(n_ch)]
            r_abs = [float(np.corrcoef(cnt[:, c], np.abs(m[:, c]))[0, 1]) for c in range(n_ch)]
            F = B.reshape(N * n_ch, -1).astype(np.float64)
            r2 = oof_r2(F, m.reshape(-1), groups)
            r2_count = oof_r2(cnt.reshape(-1, 1), m.reshape(-1), groups)
            res[f"{inp}_{wn}"] = {
                "rate": rate,
                "r_count_m": {"median": float(np.median(r_signed)), "min": min(r_signed), "max": max(r_signed),
                              "per_channel": r_signed},
                "r_count_absm": {"median": float(np.median(r_abs)), "min": min(r_abs), "max": max(r_abs),
                                 "per_channel": r_abs},
                "oof_R2_m_from_BSC6": r2, "oof_R2_m_from_count": r2_count}
            print(f"[J9 {J.DATASET} {inp} {wn}] rate={rate:.3f} r(count,m) median={np.median(r_signed):+.3f} "
                  f"[{min(r_signed):+.3f},{max(r_signed):+.3f}] r(count,|m|) median={np.median(r_abs):+.3f} "
                  f"R2(BSC6)={r2:.3f} R2(count)={r2_count:.3f} ({time.time() - t0:.0f}s)", flush=True)
    J.dump(res, "j9_tcrzem_zscore.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
