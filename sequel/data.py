"""SHAPE loaders and preprocessing.

The restricted SHAPE files never enter this repository. Paths come from environment
variables or command-line arguments:

* ``SHAPE_DIR``     three-condition directory with
                    ``SHAPE_Community_{ID}_IAPS{Neg,Neu,Pos}_BC.txt`` (1229 x 34 each)
* ``SHAPE_PICKLE``  the BIBM feature pickle (``X_ds``, ``y``, ``subjects``), used only to
                    reproduce the BIBM numbers (``experiments/check_bibm_regression.py``)

Preprocessing choices exposed here are exactly the ones the journal paper varies:

``decimation``  ``legacy`` (``raw[0:1024:4]``, no anti-alias filter) or ``fir``
                (anti-alias FIR decimation of ``raw[1:]``)
``norm``        ``epoch_z``    z-score per epoch and channel over the whole kept epoch (BIBM).
                               Pre-stimulus samples then carry an offset set by the
                               post-stimulus signal.
                ``baseline_z`` subtract the pre-stimulus mean and divide by the pre-stimulus
                               SD per epoch and channel. Pre-stimulus values depend on
                               pre-stimulus samples only.
                ``pre_center`` subtract the pre-stimulus mean, divide by the whole-epoch SD
                               (keeps epoch_z's scale normalization, removes its offset leak).
                ``global_scale`` microvolts divided by one dataset-wide SD (no per-epoch
                               scaling, so trial counts cannot rescale conditions).
                ``none``       microvolts as recorded (baseline-corrected upstream).
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from . import config as C

COND3 = {"Neg": 0, "Neu": 1, "Pos": 2}
EXCLUDE_3CLASS = {"127"}          # flat record (dissertation QC; BIBM: "one anomalous record")
_FNAME = re.compile(r"SHAPE_Community_(?P<sid>[^_]+)_IAPS(?P<cond>Neg|Neu|Pos)_BC\.txt$")


@dataclass
class Epochs:
    """Analysed signal plus labels. ``X`` is (n_obs, n_times, n_channels)."""
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    times_ms: np.ndarray
    onset: int = C.ONSET
    fs: int = C.FS
    info: dict = field(default_factory=dict)


def load_shape_raw(shape_dir: str | os.PathLike | None = None,
                   exclude: set[str] = EXCLUDE_3CLASS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read every complete three-condition subject. Returns raw (N,1229,34), y, subject ids."""
    shape_dir = Path(shape_dir or os.environ.get("SHAPE_DIR", ""))
    if not shape_dir.is_dir():
        raise SystemExit("SHAPE three-condition directory not found; set $SHAPE_DIR")
    files: dict[str, dict[int, Path]] = {}
    for p in sorted(shape_dir.iterdir()):
        m = _FNAME.match(p.name)
        if m and m["sid"] not in exclude:
            files.setdefault(m["sid"], {})[COND3[m["cond"]]] = p
    X, y, g = [], [], []
    for sid in sorted(files):
        if len(files[sid]) != 3:          # keep complete condition panels only
            continue
        for cond in sorted(files[sid]):
            X.append(np.loadtxt(files[sid][cond]))
            y.append(cond)
            g.append(sid)
    return np.asarray(X, float), np.asarray(y), np.asarray(g)


def load_bibm_pickle(path: str | os.PathLike | None = None) -> Epochs:
    """The BIBM analysed signal, unchanged (legacy decimation, epoch z-scoring)."""
    path = Path(path or os.environ.get("SHAPE_PICKLE", ""))
    if not path.is_file():
        raise SystemExit("BIBM pickle not found; set $SHAPE_PICKLE")
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    X = np.asarray(d["X_ds"], float)
    times = (np.arange(X.shape[1]) * C.DECIM - C.ONSET_RAW) / C.FS_RAW * 1000.0
    return Epochs(X, np.asarray(d["y"]), np.asarray(d["subjects"]), times,
                  info={"source": "bibm_pickle", "decimation": "legacy", "norm": "epoch_z"})


def decimate_raw(raw: np.ndarray, mode: str = "fir", fs_raw: int = C.FS_RAW,
                 onset_raw: int = C.ONSET_RAW) -> tuple[np.ndarray, np.ndarray]:
    """raw (N, T, ch) at ``fs_raw`` -> (signal at 256 Hz, times in ms relative to onset).

    ``fir`` starts at sample ``onset_raw mod q`` so that onset falls exactly on a kept
    sample (for SHAPE, q = 4 and the start is raw[1:]). ``legacy`` is the BIBM SHAPE
    signal only.
    """
    q = fs_raw // C.FS
    if q * C.FS != fs_raw:
        raise ValueError(f"fs_raw {fs_raw} is not an integer multiple of {C.FS} Hz")
    if mode == "legacy":
        if (fs_raw, onset_raw) != (C.FS_RAW, C.ONSET_RAW):
            raise ValueError("legacy decimation is defined for the SHAPE signal only")
        idx = np.arange(0, 1024, C.DECIM)
        out = raw[:, idx, :]
    elif mode == "fir":
        start = onset_raw % q
        idx = np.arange(start, raw.shape[1], q)
        out = raw[:, start:, :] if q == 1 else decimate(raw[:, start:, :], q, ftype="fir",
                                                        zero_phase=True, axis=1)
        idx = idx[: out.shape[1]]
    else:
        raise ValueError(f"unknown decimation mode {mode!r}")
    times = (idx - onset_raw) / fs_raw * 1000.0
    return np.ascontiguousarray(out), times


def normalize(X: np.ndarray, times_ms: np.ndarray, mode: str = "epoch_z") -> np.ndarray:
    """Per-epoch, per-channel normalization. ``X`` is (N, T, ch)."""
    if mode == "none":
        return X.copy()
    if mode == "global_scale":
        # One label-free scalar for the whole dataset: microvolt shapes are kept, and the
        # scale cannot differ between conditions. Per-epoch normalizations (epoch_z,
        # baseline_z, pre_center) divide each condition average by its own SD, which
        # depends on how many trials it averages (noise SD ~ 1/sqrt(n)); with unequal trial
        # counts (oddball: 40 rare vs 156 frequent) that rescales conditions differently and
        # can cancel the effect itself (ERP CORE P3 at Pz: d_z 1.05 in uV, 0.05 baseline_z).
        sd = X.std()
        return X / (sd if sd > 0 else 1.0)
    if mode == "pre_center":
        # Separates the two things epoch_z does: the pre-stimulus mean is removed (no
        # post-stimulus offset reaches the baseline) but the per-epoch scale is the
        # whole-epoch SD (the cross-subject amplitude normalization is kept).
        mu = X[:, times_ms < 0, :].mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    if mode == "epoch_z":
        ref = X
    elif mode == "baseline_z":
        ref = X[:, times_ms < 0, :]
    else:
        raise ValueError(f"unknown normalization {mode!r}")
    mu = ref.mean(axis=1, keepdims=True)
    sd = ref.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def preprocess(raw: np.ndarray, y: np.ndarray, groups: np.ndarray,
               decimation: str = "fir", norm: str = "baseline_z", fs_raw: int = C.FS_RAW,
               onset_raw: int = C.ONSET_RAW, source: str = "shape_raw") -> Epochs:
    X, times = decimate_raw(raw, decimation, fs_raw, onset_raw)
    X = normalize(X, times, norm)
    onset = int(np.argmin(np.abs(times)))
    return Epochs(X, y, groups, times, onset=onset,
                  info={"source": source, "decimation": decimation, "norm": norm})
