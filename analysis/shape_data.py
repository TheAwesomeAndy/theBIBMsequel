#!/usr/bin/env python3
"""SHAPE loader for the journal experiments (restricted data; local only, never in git).

Two local sources under $SHAPE_DIR (default /home/user/data_local/shape):
  raw/shape_features_211.pkl   the conference array: X_ds (633, 256, 34), y, subjects
                               (211 subjects x negative / neutral / pleasant);
  txt/SHAPE_Community_<ID>_IAPS<Neg|Neu|Pos>_BC.txt
                               subject-average epochs, 1229 x 34 in microvolts at
                               1024 Hz, -200..+1000 ms (onset at row 205), baseline-
                               corrected; one more subject than the pickle.
The conference input is X_ds == zscore(raw[0:1024:4]) per epoch and channel: every
fourth row of rows 0..1023 (256 samples at 256 Hz, -200..+797 ms, onset at sample 51,
no anti-alias filter), z-scored within the epoch. `verify()` checks that identity on
every observation and fixes the label coding from it (labels 0, 1, 2 are matched to
files by value, not assumed).

load_cohort(zscore=True)  -> X_ds exactly as the conference scripts read it;
load_cohort(zscore=False) -> the same 633 observations in microvolts (raw[0:1024:4]),
                             same order, for analyses that need the unnormalized signal.
The pickle is read with an unpickler restricted to numpy array reconstruction.
The channel order is not documented, so no montage-based analysis runs on SHAPE.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

SHAPE_DIR = Path(os.environ.get("SHAPE_DIR", "/home/user/data_local/shape"))
PKL = SHAPE_DIR / "raw" / "shape_features_211.pkl"
TXT = SHAPE_DIR / "txt"
CACHE = SHAPE_DIR / "cache"
FS_IN, ONSET_ROW, DECIM, N_SAMPLES = 1024, 205, 4, 256
VALENCES = ("Neg", "Neu", "Pos")


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") and name in ("_reconstruct", "ndarray", "dtype", "scalar"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_pickle():
    with open(PKL, "rb") as f:
        d = _NumpyOnly(f).load()
    return np.asarray(d["X_ds"], float), np.asarray(d["y"]), np.asarray(d["subjects"])


def zscore_epochs(X):
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _txt(sid, val):
    return TXT / f"SHAPE_Community_{sid:03d}_IAPS{val}_BC.txt"


def onset_sample() -> int:
    return ONSET_ROW // DECIM              # sample 51 = row 204 (-0.8 ms), the conference onset


def load_uv():
    """(633, 256, 34) microvolt epochs in pickle order, plus the label->valence coding."""
    cache = CACHE / "uv.npy"
    X, y, g = load_pickle()
    if cache.exists():
        U = np.load(cache)
    else:
        U = np.zeros_like(X)
        for i, (lab, sid) in enumerate(zip(y, g)):
            # the valence of each pickle label is fixed by verify(); default order Neg/Neu/Pos
            raw = np.loadtxt(_txt(int(sid), VALENCES[int(lab)]))
            U[i] = raw[0:FS_IN:DECIM]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, U)
    return U, y, g


def verify(verbose=True):
    """Check X_ds == zscore(raw[0:1024:4]) for every observation under the label coding
    0/1/2 = Neg/Neu/Pos, and that no other coding fits. Returns the max abs deviation."""
    X, y, g = load_pickle()
    worst = 0.0
    for i, (lab, sid) in enumerate(zip(y, g)):
        dev = {}
        for v in VALENCES:
            raw = np.loadtxt(_txt(int(sid), v))
            dev[v] = float(np.abs(zscore_epochs(raw[None, 0:FS_IN:DECIM])[0] - X[i]).max())
        best = min(dev, key=dev.get)
        if best != VALENCES[int(lab)]:
            raise AssertionError(f"observation {i}: label {lab} matches {best}, not {VALENCES[int(lab)]}")
        worst = max(worst, dev[best])
    if verbose:
        print(f"[shape] verified {len(y)} observations, {len(np.unique(g))} subjects; "
              f"labels 0/1/2 = Neg/Neu/Pos; max |X_ds - zscore(raw[0:1024:4])| = {worst:.2e}")
    return worst


def load_cohort(zscore=True):
    if zscore:
        return load_pickle()
    return load_uv()


if __name__ == "__main__":
    verify()
