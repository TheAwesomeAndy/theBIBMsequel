#!/usr/bin/env python3
"""Signal-level perturbations shared by every encoder (camera-ready, reviewer R2.1).

All perturbations act on the analysed signal X (n_obs, 256 samples, 34 channels) BEFORE
any feature extraction, so band-power, ERP-window, the reservoir and EEGNet are corrupted
identically. Random draws are keyed to (seed, fold) so that every encoder sees the same
removed electrodes, the same noise realization, and the same timing shifts.

- remove_channels: removed electrodes become zero traces. The draw reproduces
  analysis/experiment2_rawsignal.py exactly (np.random.default_rng(1000*seed + fold)).
- amplitude_noise: additive white Gaussian noise at a fixed signal-to-noise ratio,
  computed per observation and channel from the signal's own power.
- temporal_jitter: each observation is shifted by an integer offset drawn uniformly
  from +/- max_ms (the same offset on all channels, i.e. an alignment error), with
  edge-value padding instead of wrap-around.
"""
from __future__ import annotations

import numpy as np

FS = 256  # Hz


def removal_draw(n_ch: int, frac: float, seed: int, fold: int) -> np.ndarray:
    if frac <= 0:
        return np.array([], int)
    rng = np.random.default_rng(1000 * seed + fold)
    return rng.choice(n_ch, int(round(frac * n_ch)), replace=False)


def remove_channels(X: np.ndarray, frac: float, seed: int, fold: int):
    drop = removal_draw(X.shape[2], frac, seed, fold)
    Xc = X.copy()
    Xc[:, :, drop] = 0.0
    return Xc, drop


def amplitude_noise(X: np.ndarray, snr_db: float, seed: int, fold: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + 1000 * seed + fold)
    power = (X ** 2).mean(axis=1, keepdims=True)                 # (n, 1, ch)
    sd = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return X + sd * rng.standard_normal(X.shape)


def temporal_jitter(X: np.ndarray, max_ms: float, seed: int, fold: int) -> np.ndarray:
    rng = np.random.default_rng(20_000 + 1000 * seed + fold)
    k = int(round(max_ms / 1000.0 * FS))
    shifts = rng.integers(-k, k + 1, X.shape[0])
    out = np.empty_like(X)
    T = X.shape[1]
    for i, s in enumerate(shifts):
        if s >= 0:                                                # delay: pad the start
            out[i] = np.concatenate([np.repeat(X[i, :1], s, axis=0), X[i, :T - s]], axis=0)
        else:                                                     # advance: pad the end
            out[i] = np.concatenate([X[i, -s:], np.repeat(X[i, -1:], -s, axis=0)], axis=0)
    return out


SIGNAL_CONDITIONS = {
    # name -> function(X, seed, fold) returning the corrupted signal
    "amp_5dB": lambda X, seed, fold: amplitude_noise(X, 5.0, seed, fold),
    "jitter_50ms": lambda X, seed, fold: temporal_jitter(X, 50.0, seed, fold),
}
