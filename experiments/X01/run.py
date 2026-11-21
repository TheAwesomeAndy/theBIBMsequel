"""X01-2026-11: does per-epoch normalization create pre-stimulus decoding on SHAPE?

Theory (paper Sec. III): per-epoch z-scoring subtracts the epoch mean, which
contains the post-stimulus evoked response, from every sample. A pre-stimulus
sample therefore carries -E[m|y]/sd, a condition-dependent constant. Scaling
that uses no post-stimulus sample cannot leak.

Prediction, fixed before this run:
  P1  Every encoder restricted to pre-stimulus samples decodes above chance
      under epoch_z and post_z, and at chance under pre_sd, none, global_sd.
  P2  The BIBM reservoir (window -160..+73 ms) reproduces BA 0.487 under
      epoch_z and loses most of its above-chance accuracy under pre_sd.
Anchor: BIBM reports reservoir BA 0.487 [0.447, 0.523] under epoch_z.

Usage: SHAPE_DATA_DIR=... python experiments/X01/run.py
Output: outputs/X01/x01_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

NORMS = ["epoch_z", "post_z", "pre_sd", "none", "global_sd"]
PRE_LO = 9                       # pre-stimulus analysis window [9, 51): -164..0 ms
OUT = ROOT / "outputs" / "X01" / "x01_results.json"


def pre_bins(X, n_bins=6):
    seg = X[:, PRE_LO:cm.ONSET_DS, :]
    w = seg.shape[1] // n_bins
    return np.concatenate([seg[:, i * w:(i + 1) * w, :].mean(1) for i in range(n_bins)], axis=1)


def erp_windows(X):
    # BIBM windows in sample-index ms from epoch start (80-200, 250-450, 450-800)
    wins = [(80, 200), (250, 450), (450, 800)]
    return np.concatenate([X[:, int(round(a / 1000 * 256)):int(round(b / 1000 * 256)), :].mean(1)
                           for a, b in wins], axis=1)


def main():
    t0 = time.time()
    raw, y, g = cm.load_shape_raw()
    Xb = cm.downsample(raw)
    print(f"loaded {raw.shape}, subjects {len(np.unique(g))}, analysed {Xb.shape}", flush=True)
    preds, rates = {}, {}
    for norm in NORMS:
        X = cm.normalize(Xb, norm)
        feats = {
            "pre_mean": X[:, PRE_LO:cm.ONSET_DS, :].mean(1),
            "pre_bins6": pre_bins(X),
            "erp_windows": erp_windows(X),
        }
        for name, F in feats.items():
            preds[f"{name}|{norm}"] = cm.oof_proba(F, y, g).argmax(1)
        for name, (a, b) in {"reservoir_bibm": (10, 70), "reservoir_pre": (PRE_LO, cm.ONSET_DS)}.items():
            Fb, rate = cm.reservoir_bsc(X, t_start=a, t_end=b)
            rates[f"{name}|{norm}"] = rate
            preds[f"{name}|{norm}"] = cm.oof_proba(Fb, y, g, block_pca_k=64).argmax(1)
            print(f"  {name}|{norm}: rate {rate:.3f}, elapsed {time.time() - t0:.0f}s", flush=True)
    summary, B, names = cm.subject_bootstrap(y, g, preds)
    contrasts = {}
    for name in ["pre_mean", "pre_bins6", "erp_windows", "reservoir_bibm", "reservoir_pre"]:
        for norm in NORMS[1:]:
            contrasts[f"{name}: epoch_z - {norm}"] = cm.paired(B, names, f"{name}|epoch_z", f"{name}|{norm}")
    null = {}
    for norm in ["epoch_z", "pre_sd"]:
        F = cm.normalize(Xb, norm)[:, PRE_LO:cm.ONSET_DS, :].mean(1)
        d = cm.label_permutation_null(F, y, g, n_perm=200)
        null[f"pre_mean|{norm}"] = {"mean": float(d.mean()), "sd": float(d.std()),
                                    "q95": float(np.percentile(d, 95)), "max": float(d.max())}
    res = {
        "provenance": cm.provenance("X01-2026-11", {
            "norms": NORMS, "pre_window_ds": [PRE_LO, cm.ONSET_DS],
            "reservoir": {"N": cm.N_RES, "beta": cm.BETA, "theta": cm.THETA, "rho": 0.9,
                          "weight_seed": cm.W_SEED, "pca_k": 64},
            "protocol": "StratifiedGroupKFold(5, shuffle) seeds 42-46, train-only scaler, "
                        "balanced L2 logistic, OOF probabilities summed, 4000 subject bootstraps",
            "null": "200 within-subject label permutations, seed 42 partition only"}),
        "n_obs": int(len(y)), "n_subjects": int(len(np.unique(g))),
        "balanced_accuracy": summary, "contrasts": contrasts,
        "firing_rate": rates, "permutation_null": null,
        "runtime_s": round(time.time() - t0, 1),
    }
    cm.write_json(OUT, res)
    for k, v in summary.items():
        print(f"{k:32s} {v['ba']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    print(null)


if __name__ == "__main__":
    main()
