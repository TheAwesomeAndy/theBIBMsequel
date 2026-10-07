#!/usr/bin/env python3
"""J8: how far a silent electrode lies from typical reservoir activity, per operating point.

A removed electrode is a silent drive, and a silent drive is a fixed point of the
reservoir: its spike-count block is exactly zero, which the train-only PCA maps to
v0 = -P b_bar. In the standardized readout coordinates this block sits at
z0_k = (v0_k - mu_k) / sigma_k for every component k, the same quantity that sets the
zero-fill shift of Proposition 1. This script measures, for reservoir draw 42 on the
selected epoch and rho in {0, 0.3, 0.6, 0.9, 1.2, 1.5}:
  * the mean firing rate in the analysis window;
  * mean_k |z0_k| for the silent-electrode block (training subjects of each fold,
    averaged over the 25 folds), and
  * mean |mu|/sigma of the raw spike counts (before PCA).
It complements J4, which reports BA clean and with 30% of the electrodes silent.
Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402

RHOS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]


def main():
    t0 = time.time()
    X, y, g = J.load_cohort()
    N, T, n_ch = X.shape
    F = J.folds(y, g)
    res = {"epoch": J.EPOCH, "draw": 42, "rho": RHOS, "rate": {}, "silent_block_mean_abs_z": {},
           "raw_counts_mean_abs_mu_over_sigma": {},
           "protocol": "reservoir draw 42; train-only PCA-64 per fold (StratifiedGroupKFold(5) x seeds "
                       "42-46); silent block v0 = PCA.transform(0); z0 = (v0 - mean)/sd of the training "
                       "embedding; averaged over components and folds"}
    for rho in RHOS:
        B, rate = J.reservoir_codes(X, rho=rho)
        D = B.shape[2]
        zs, cs = [], []
        for seed, fold, tr, te in F:
            pca = PCA(n_components=J.PCA_K, random_state=J.W_SEED).fit(B[tr].reshape(-1, D))
            Etr = pca.transform(B[tr].reshape(-1, D))           # (len(tr)*ch, 64)
            v0 = pca.transform(np.zeros((1, D)))[0]
            E3 = Etr.reshape(len(tr), n_ch, -1)                  # readout features are (obs, ch*64)
            mu = E3.mean(0); sd = E3.std(0) + 1e-12              # per (channel, component)
            zs.append(float(np.mean(np.abs((v0[None, :] - mu) / sd))))
            R = B[tr].reshape(len(tr), -1).astype(np.float64)
            rs = R.std(0); ok = rs > 1e-12
            cs.append(float(np.mean(np.abs(R.mean(0)[ok]) / rs[ok])))
        res["rate"][str(rho)] = rate
        res["silent_block_mean_abs_z"][str(rho)] = float(np.mean(zs))
        res["raw_counts_mean_abs_mu_over_sigma"][str(rho)] = float(np.mean(cs))
        print(f"[J8 rho={rho}] rate={rate:.3f} |z0|={np.mean(zs):.2f} raw |mu|/sigma={np.mean(cs):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    J.dump(res, "j8_tcrzem_silence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
