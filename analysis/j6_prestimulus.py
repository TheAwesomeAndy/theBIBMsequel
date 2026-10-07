#!/usr/bin/env python3
"""J6: what the encoders read from the pre-stimulus interval (external TCRZEM cohort).

The conference reservoir window t in [10, 70) spans about -160..+73 ms around onset
(sample 51), so most of it precedes the stimulus (docs/CODE_AUDIT.md, A18). Before
onset a trial-averaged, baseline-corrected ERP carries no stimulus-locked signal, but
the conference input is z-scored per channel over the whole epoch, which subtracts the
epoch mean and divides by the epoch SD: both are set mostly by the post-stimulus signal,
so the pre-stimulus samples inherit an offset and scale that depend on the response.

Two input normalizations:
  Z  per-epoch, per-channel z-score over the full epoch (conference convention);
  G  baseline-corrected microvolts divided by one fixed scale per channel (the channel's
     SD pooled over all observations and samples; 32 numbers, label-free, applied
     identically to every epoch), so no information crosses from the post- to the
     pre-stimulus interval within an epoch. The readout standardization stays train-only.
Encoders and windows (sample index, onset 51):
  Reservoir (rho = 0.9, BSC6, train-only PCA-64): pre [3, 51) (-188..0 ms),
     conference [10, 70), early-post [51, 111) (0..+234 ms), post [51, 255) (0..+797 ms);
  ERP-window: conference (three windows), pre-only (-200..0 ms), post-only (two windows).
Reported: subject-level BA [95% CI] for every (encoder window, normalization), paired
Z minus G, and the correlation between the pre-stimulus mean of the z-scored input and
the post-stimulus mean of the baseline-corrected signal across observations and channels.
Protocol and inference as in jcore.py. Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import tcrzem_data as TD  # noqa: E402

ONSET = 51
RES_WINDOWS = {"pre": (3, 51), "conference": (10, 70), "early_post": (51, 111), "post": (51, 255)}
if J.EPOCH == "long":                        # pre-specified long epoch: add 0..+2484 ms
    RES_WINDOWS["post_long"] = (51, 51 + 6 * 106)
# sample-index milliseconds from the first sample (-200 ms)
ERP_SETS = {"conference": [(80, 200), (250, 450), (450, 800)], "pre_only": [(0, 200)], "post_only": [(250, 450), (450, 800)]}
if J.EPOCH == "long":
    ERP_SETS["post_only_long"] = [(250, 450), (450, 800), (700, 1500), (1700, 2700)]


def reservoir_multiwindow(X, windows, n_bins=6, rho=J.RHO, seed=J.W_SEED):
    """One reservoir pass; BSC codes for several windows: {name: (N, ch, 256*n_bins)}."""
    N, T, n_ch = X.shape
    S = np.transpose(X, (1, 0, 2)).reshape(T, N * n_ch)
    W_in, W = J.reservoir_weights(rho, seed); win0 = W_in[:, 0][:, None]
    C = S.shape[1]
    mem = np.zeros((J.N_RES, C)); spk = np.zeros((J.N_RES, C))
    counts = {k: np.zeros((J.N_RES, n_bins, C), dtype=np.float32) for k in windows}
    t_max = max(b for _, b in windows.values())
    for t in range(t_max):
        mem = (1.0 - J.BETA) * mem * (1.0 - spk) + win0 * S[t][None, :] + W @ spk
        spk = (mem >= J.THETA).astype(np.float64)
        mem = np.maximum(mem - spk * J.THETA, 0.0)
        for k, (a, b) in windows.items():
            bw = (b - a) // n_bins
            if a <= t < a + bw * n_bins:
                counts[k][:, (t - a) // bw, :] += spk
    return {k: v.reshape(J.N_RES * n_bins, C).T.reshape(N, n_ch, -1) for k, v in counts.items()}


def main():
    t0 = time.time()
    Xuv, y, g = TD.load_cohort(zscore=False)
    Xz = TD.zscore_epochs(Xuv)
    N, T, n_ch = Xz.shape
    classes = np.unique(y)
    F = J.folds(y, g)
    variants = [f"Reservoir:{w}" for w in RES_WINDOWS] + [f"ERP-window:{w}" for w in ERP_SETS]
    acc = {(v, nm): np.zeros((N, classes.size)) for v in variants for nm in ("Z", "G")}
    codes_z = reservoir_multiwindow(Xz, RES_WINDOWS)
    rate = {k: float(v.mean() / ((RES_WINDOWS[k][1] - RES_WINDOWS[k][0]) // 6)) for k, v in codes_z.items()}
    print(f"[J6] Z codes ready ({time.time() - t0:.0f}s)", flush=True)
    Xg = Xuv / Xuv.std(axis=(0, 1))                                # one fixed scale per channel
    codes_g = reservoir_multiwindow(Xg, RES_WINDOWS)
    print(f"[J6] G codes ready ({time.time() - t0:.0f}s)", flush=True)
    for seed, fold, tr, te in F:
        for nm, Xin, codes in (("Z", Xz, codes_z), ("G", Xg, codes_g)):
            for w in RES_WINDOWS:
                _, E, _ = J.pca_embed(codes[w], tr)
                Zf = E.reshape(N, -1)
                acc[(f"Reservoir:{w}", nm)][te] += J.fit_pred(Zf[tr], y[tr], Zf[te], seed)
            for w, wins in ERP_SETS.items():
                Zf = J.erp_windows(Xin, wins).reshape(N, -1)
                acc[(f"ERP-window:{w}", nm)][te] += J.fit_pred(Zf[tr], y[tr], Zf[te], seed)
        print(f"[J6] seed {seed} fold {fold} ({time.time() - t0:.0f}s)", flush=True)
    preds = {k: J.argmax_pred(v, classes) for k, v in acc.items()}
    pre_z = Xz[:, :ONSET, :].mean(axis=1).ravel()
    post_uv = Xuv[:, ONSET:, :].mean(axis=1).ravel()
    res = {"protocol": "Z = per-epoch per-channel z-score over the full epoch; G = baseline-corrected "
                       "microvolts / one fixed per-channel SD (pooled, label-free); reservoir rho=0.9 BSC6 per window, "
                       "train-only PCA-64; StratifiedGroupKFold(5) x seeds 42-46; balanced L2 logreg; "
                       "OOF proba averaged over partitions; subject-level bootstrap n_boot=%d" % J.N_BOOT,
           "reservoir_windows_samples": RES_WINDOWS, "erp_sets_sample_ms": ERP_SETS,
           "corr_prestim_mean_Z_vs_poststim_mean_uV": float(np.corrcoef(pre_z, post_uv)[0, 1]),
           "firing_rate_Z": rate, "BA": {}, "paired_Z_minus_G": {}}
    for v in variants:
        pt, ci, _ = J.boot_ci(y, g, {nm: preds[(v, nm)] for nm in ("Z", "G")})
        res["BA"][v] = {nm: {"BA": pt[nm], "ci95": ci[nm]} for nm in ("Z", "G")}
        res["paired_Z_minus_G"][v] = J.paired_ci(y, g, preds[(v, "Z")], preds[(v, "G")])
        print(f"[J6 {v}] Z={pt['Z']:.3f}[{ci['Z'][0]:.3f},{ci['Z'][1]:.3f}] "
              f"G={pt['G']:.3f}[{ci['G'][0]:.3f},{ci['G'][1]:.3f}]", flush=True)
    res["paired_post_minus_conference_reservoir"] = {
        nm: J.paired_ci(y, g, preds[("Reservoir:post", nm)], preds[("Reservoir:conference", nm)]) for nm in ("Z", "G")}
    print(f"[J6] corr(pre-stim mean Z, post-stim mean uV) = {res['corr_prestim_mean_Z_vs_poststim_mean_uV']:.3f}")
    J.dump(res, "j6_tcrzem_prestimulus.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
