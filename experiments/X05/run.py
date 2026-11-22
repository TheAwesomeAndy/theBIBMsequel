"""X05-2026-11: time-resolved decoding on SHAPE under each scaling.

Per 4-sample bin (15.6 ms at 256 Hz) of the analysed epoch, the 34-channel
pattern is decoded with the BIBM protocol (StratifiedGroupKFold 5, seeds
42-46, train-only standardization, balanced L2 logistic, out-of-fold
probabilities summed over partitions). Per-bin 95% intervals from 1000
participant bootstrap resamples.

Prediction, fixed before this run: under epoch_z the decoding curve is above
chance throughout the pre-stimulus interval, and under pre_sd and none it
is at chance before onset and rises after it.

Usage: SHAPE_DATA_DIR=... python experiments/X05/run.py
Output: outputs/X05/x05_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

NORMS = ["epoch_z", "pre_sd", "none"]
B = 4
OUT = ROOT / "outputs" / "X05" / "x05_results.json"


def boot_ba(y, g, correct_by_bin, n_boot=1000, seed=0):
    subs = np.unique(g); idx = [np.flatnonzero(g == s) for s in subs]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, correct_by_bin.shape[1]))
    for b in range(n_boot):
        rows = np.concatenate([idx[i] for i in rng.integers(0, len(subs), len(subs))])
        yy = y[rows]; cc = correct_by_bin[rows]
        out[b] = np.mean([cc[yy == k].mean(0) for k in np.unique(y)], axis=0)
    return np.percentile(out, [2.5, 97.5], axis=0)


def main():
    t0 = time.time()
    raw, y, g = cm.load_shape_raw()
    X = cm.downsample(raw)
    nb = X.shape[1] // B
    centers = ((np.arange(nb) * B + (B - 1) / 2) - cm.ONSET_DS) / cm.FS_DS * 1000
    res = {"bin_center_ms": centers.round(2).tolist()}
    for n in NORMS:
        Xb = cm.normalize(X, n)[:, :nb * B].reshape(len(X), nb, B, -1).mean(2)
        correct = np.zeros((len(y), nb), bool)
        for j in range(nb):
            correct[:, j] = cm.oof_proba(Xb[:, j], y, g).argmax(1) == y
        ba = np.mean([correct[y == k].mean(0) for k in range(3)], axis=0)
        lo, hi = boot_ba(y, g, correct)
        pre = centers < 0
        res[n] = {"ba": ba.round(4).tolist(), "ci_lo": lo.round(4).tolist(), "ci_hi": hi.round(4).tolist(),
                  "pre_mean_ba": float(ba[pre].mean()),
                  "first_bin_ci_above_chance_ms": (float(centers[np.argmax(lo > 1 / 3)]) if (lo > 1 / 3).any() else None)}
        print(n, "pre-stim mean BA", round(res[n]["pre_mean_ba"], 3), "first CI>chance at",
              res[n]["first_bin_ci_above_chance_ms"], "ms", f"{time.time() - t0:.0f}s", flush=True)
    cm.write_json(OUT, {"provenance": cm.provenance("X05-2026-11", {"norms": NORMS, "bin_samples": B,
                                                                    "n_boot": 1000}), **res})


if __name__ == "__main__":
    main()
