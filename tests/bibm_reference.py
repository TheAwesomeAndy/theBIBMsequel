"""Verbatim BIBM reference implementations, used only as test oracles.

Copied from TheAwesomeAndy/BIBM@1667277 ``analysis/experiment_camera_ready.py`` (``weights``,
``bsc6``) and ``analysis/experiment2_rawsignal.py`` (``bandpower_all``) with constants
inlined. Do not edit: the tests prove the new package reproduces these exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

N_RES, BETA, THETA, SEED = 256, 0.05, 0.5, 42
BSC_N_BINS, T_START, T_END = 6, 10, 70
FS = 256
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def weights(rho):
    rng = np.random.RandomState(SEED)
    lin = np.sqrt(6.0 / (1 + N_RES)); W_in = rng.uniform(-lin, lin, (N_RES, 1))
    lrec = np.sqrt(6.0 / (2 * N_RES)); W = rng.uniform(-lrec, lrec, (N_RES, N_RES))
    W *= rho / np.abs(np.linalg.eigvals(W)).max()
    return W_in, W


def bsc6(X_ds, rho):
    N, T, n_ch = X_ds.shape; C = N * n_ch
    S = np.transpose(X_ds, (1, 0, 2)).reshape(T, C)
    W_in, W = weights(rho); win0 = W_in[:, 0][:, None]
    mem = np.zeros((N_RES, C)); spk = np.zeros((N_RES, C))
    counts = np.zeros((N_RES, BSC_N_BINS, C), dtype=np.float32)
    binw = (T_END - T_START) // BSC_N_BINS
    for t in range(T):
        mem = (1.0 - BETA) * mem * (1.0 - spk) + win0 * S[t][None, :] + W @ spk
        spk = (mem >= THETA).astype(np.float64)
        mem = np.maximum(mem - spk * THETA, 0.0)
        if T_START <= t < T_END:
            counts[:, (t - T_START) // binw, :] += spk
    rate = float(counts.sum() / (N_RES * C * (T_END - T_START)))
    return counts.reshape(N_RES * BSC_N_BINS, C).T.reshape(N, n_ch, -1), rate


def bandpower_all(X_ds):
    N, T, n_ch = X_ds.shape
    out = np.zeros((N, n_ch, 5))
    for i in range(N):
        for ch in range(n_ch):
            f, p = welch(X_ds[i, :, ch], fs=FS, nperseg=min(T, 256))
            for bi, (lo, hi) in enumerate(BANDS):
                m = (f >= lo) & (f <= hi)
                if m.any():
                    out[i, ch, bi] = np.trapezoid(p[m], f[m])
    return out
