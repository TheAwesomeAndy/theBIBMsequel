"""Fixed feature encoders. Every function returns per-channel blocks (n_obs, n_ch, n_feat)."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA

from . import config as C


def bandpower(X: np.ndarray, fs: int = C.FS, bands=C.BANDS) -> np.ndarray:
    """Welch band power per channel (BIBM estimator: nperseg = min(T, 256), trapezoid)."""
    n_obs, T, n_ch = X.shape
    f, p = welch(X, fs=fs, nperseg=min(T, 256), axis=1)       # p: (n_obs, n_freq, n_ch)
    out = np.zeros((n_obs, n_ch, len(bands)))
    for b, (lo, hi) in enumerate(bands):
        m = (f >= lo) & (f <= hi)
        if m.any():
            out[:, :, b] = np.trapezoid(p[:, m, :], f[m], axis=1)
    return out


def erp_windows_legacy(X: np.ndarray, fs: int = C.FS) -> np.ndarray:
    """BIBM ERP-window features (sample-index windows on the legacy signal)."""
    out = np.zeros((X.shape[0], X.shape[2], len(C.LEGACY_ERP_WINDOWS_MS)))
    for w, (a, b) in enumerate(C.LEGACY_ERP_WINDOWS_MS):
        out[:, :, w] = X[:, int(round(a / 1000 * fs)):int(round(b / 1000 * fs)), :].mean(axis=1)
    return out


def erp_windows(X: np.ndarray, times_ms: np.ndarray, windows=C.ERP_WINDOWS_REL_MS) -> np.ndarray:
    """Mean amplitude per channel in windows given in ms relative to onset, [a, b)."""
    out = np.zeros((X.shape[0], X.shape[2], len(windows)))
    for w, (a, b) in enumerate(windows):
        m = (times_ms >= a) & (times_ms < b)
        if not m.any():
            raise ValueError(f"empty ERP window {(a, b)}")
        out[:, :, w] = X[:, m, :].mean(axis=1)
    return out


def amplitude_bins(X: np.ndarray, window: tuple[int, int], n_bins: int = 6) -> np.ndarray:
    """Matched non-spiking control: mean amplitude in the reservoir's own bins.

    Same window and bin edges as ``LIFReservoir.encode``, so any accuracy the reservoir adds
    over this control is attributable to the spiking state-space expansion, not to binning.
    """
    t0, t1 = window
    binw = (t1 - t0) // n_bins
    out = np.zeros((X.shape[0], X.shape[2], n_bins))
    for k in range(n_bins):
        out[:, :, k] = X[:, t0 + k * binw:t0 + (k + 1) * binw, :].mean(axis=1)
    return out


class BlockPCA:
    """One PCA basis shared by all electrodes, fitted on training observations only."""

    def __init__(self, k: int = C.PCA_K, random_state: int = C.WEIGHT_SEED):
        self.k, self.random_state = k, random_state

    def fit(self, B: np.ndarray) -> "BlockPCA":
        self.pca = PCA(n_components=self.k, random_state=self.random_state).fit(
            B.reshape(-1, B.shape[2]))
        return self

    def transform(self, B: np.ndarray) -> np.ndarray:
        n_obs, n_ch, D = B.shape
        return self.pca.transform(B.reshape(-1, D)).reshape(n_obs, n_ch, self.k)

    def silent_channel(self) -> np.ndarray:
        """Embedding of an all-zero (silent) block: where a removed electrode lands."""
        return self.pca.transform(np.zeros((1, self.pca.n_features_in_)))[0]
