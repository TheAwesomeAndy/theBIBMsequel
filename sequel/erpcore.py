"""ERP CORE loader (Kappenman et al., NeuroImage 2021).

Reads the epoched, ICA-corrected, artifact-flagged single trials
(`*_epoch_interp_ar.set/.fdt`, 256 Hz, -200..800 ms, baseline-corrected over
the pre-stimulus interval). Trials flagged in EEG.reject.rejmanual are dropped.
The 28 scalp channels are kept. The data directory is read from
ERPCORE_DATA_DIR, with one subdirectory per paradigm.

Only the five stimulus-locked paradigms are used. In the response-locked ERN
and LRP epochs, activity before the event is genuine preparation, so a
pre-event interval is not a null region.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

N_SCALP = 28
# Two-class contrasts, matched on lowercase prefixes of the ERPLAB bin descriptions.
CONTRASTS = {
    "N170": (("faces, correct",), ("cars, correct",)),
    "P3": (("rare, correct",), ("frequent, correct",)),
    "N400": (("unrelated",), ("related",)),
    "N2pc": (("left",), ("right",)),
    "MMN": (("deviant",), ("standard",)),
}


def data_dir() -> Path:
    d = os.environ.get("ERPCORE_DATA_DIR")
    if not d:
        raise SystemExit("Set ERPCORE_DATA_DIR to the ERP CORE download directory.")
    return Path(d)


def participants(paradigm: str):
    pat = re.compile(r"^(\d+)_.*_epoch_interp_ar\.set$")
    out = []
    for f in sorted((data_dir() / paradigm).iterdir()):
        m = pat.match(f.name)
        if m and (f.with_suffix(".fdt")).exists():
            out.append((int(m.group(1)), f))
    return sorted(out)


def _bin_index(labels, keys):
    hits = [i + 1 for i, lab in enumerate(labels) if any(lab.startswith(k) for k in keys)]
    if len(hits) != 1:
        raise ValueError(f"bin match for {keys} in {labels}: {hits}")
    return hits[0]


def load(path_set: Path, paradigm: str):
    """Return trials (n, T, 28) in microvolts, times (ms), binary labels for the contrast."""
    m = sio.loadmat(path_set, squeeze_me=True, struct_as_record=False)["EEG"]
    x = np.fromfile(path_set.with_suffix(".fdt"), dtype="<f4").reshape(m.trials, m.pnts, m.nbchan)
    x = x[:, :, :N_SCALP].astype(np.float64)
    times = 1000.0 * (m.xmin + np.arange(m.pnts) / m.srate)
    labels = [str(b.description).strip().lower() for b in np.atleast_1d(m.EVENTLIST.bdf)]
    bini = np.array([int(np.atleast_1d(np.atleast_1d(e.eventbini)[0])[0]) for e in np.atleast_1d(m.epoch)])
    rej = (np.asarray(m.reject.rejmanual, dtype=bool) if np.size(m.reject.rejmanual)
           else np.zeros(m.trials, bool))
    a, b = (_bin_index(labels, k) for k in CONTRASTS[paradigm])
    keep = (~rej) & np.isin(bini, [a, b])
    return x[keep], times, (bini[keep] == a).astype(int), labels


def normalize(x, times, mode: str):
    """Per-trial, per-channel scaling. x is (n, T, C), baseline-corrected."""
    pre, post = times < 0, times >= 0
    if mode == "none":
        return x
    if mode == "pre_sd":
        return x / x[:, pre].std(axis=1, keepdims=True)
    if mode == "epoch_z":
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    if mode == "post_z":
        return (x - x[:, post].mean(1, keepdims=True)) / x[:, post].std(1, keepdims=True)
    raise ValueError(mode)
