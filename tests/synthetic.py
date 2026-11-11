"""Synthetic stand-in for SHAPE: subjects x 3 conditions, 34 channels, ERP-like signals.

Each observation = subject-specific waveform offset + condition-specific late deflection on a
subset of channels + noise. Used only to exercise code paths; numbers carry no meaning.
"""
from __future__ import annotations

import numpy as np


def raw_epochs(n_subj: int = 30, n_ch: int = 34, n_raw: int = 1229, seed: int = 0):
    """Raw-like epochs at 1024 Hz with onset at sample 205 and a zero-mean baseline."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n_raw) - 205) / 1024.0
    late = np.exp(-0.5 * ((t - 0.5) / 0.15) ** 2) * (t > 0)
    early = np.exp(-0.5 * ((t - 0.15) / 0.04) ** 2) * (t > 0)
    X, y, g = [], [], []
    for s in range(n_subj):
        subj = rng.normal(0, 1, n_ch)
        topo = rng.normal(0, 1, n_ch)
        for c in range(3):
            amp = np.array([1.5, 0.0, 1.0])[c]
            sig = (subj[None, :] * early[:, None] + amp * topo[None, :] * late[:, None]
                   + 0.8 * rng.normal(0, 1, (n_raw, n_ch)))
            sig -= sig[:205].mean(axis=0, keepdims=True)          # baseline correction
            X.append(sig)
            y.append(c)
            g.append(f"S{s:03d}")
    return np.asarray(X), np.asarray(y), np.asarray(g)


def electrode_positions(n_ch: int = 34, seed: int = 1) -> np.ndarray:
    """Points spread over the upper hemisphere of the unit sphere (a stand-in montage)."""
    i = np.arange(n_ch) + 0.5
    phi = np.arccos(1 - i / n_ch)            # polar angle in (0, pi/2)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
