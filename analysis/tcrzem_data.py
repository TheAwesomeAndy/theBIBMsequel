#!/usr/bin/env python3
"""External cohort loader: DataVerseNL doi:10.34894/TCRZEM (Revers et al., 2025).

IAPS passive viewing, 239 recorded participants, of whom 228 have a complete
negative / neutral / positive panel in the preprocessed export. The export is
BrainVision segmented files, one per subject x valence x arousal
(S<id>_Mastoid_<Neg|Neu|Pos>_<High|Low>.{vhdr,vmrk,dat}): INT_16, multiplexed,
41 channels (32 scalp electrodes of the BioSemi 32 layout, M1, M2, four EOG, two
ECG, Status), 1792-point segments at 512 Hz from -1000 to +2500 ms
(stimulus onset at segment point 512, 0-indexed). The upstream pipeline
re-referenced to the mastoids, filtered, corrected ocular artifacts with ICA and
removed rejected segments before export.

To match the SHAPE observations of the conference paper exactly, every
(subject, valence) observation is built as follows:
  1. keep the 32 scalp channels (M1/M2, EOG, ECG and Status are dropped);
  2. average all retained single trials of that valence (both arousal levels);
  3. cut -200 .. +800 ms (512 points), baseline-correct over -200 .. 0 ms;
  4. keep every second point (no anti-alias filter, as for SHAPE's every fourth
     point): 256 samples at 256 Hz with onset at sample 51;
  5. z-score each channel within the epoch.
Steps 3-5 reproduce the SHAPE layout used by every encoder (X_ds: 256 x channels,
onset at sample 51, per-epoch per-channel z-scoring).

Single-trial epochs (after steps 1, 3 and 4, before averaging and z-scoring) are
cached under $TCRZEM_CACHE so the trial-count experiment can re-average subsets.

$TCRZEM_EPOCH selects the epoch: "standard" (default, the conference layout above) or
"long", the pre-specified secondary analysis, which keeps the same sampling, baseline
and onset (sample 51) but extends the epoch to +2500 ms (691 samples), because the
dataset's publication reports affective salience effects at 500-1300 ms and valence
effects at 1500-2500 ms after onset.

Data never enter the repository. $TCRZEM_DIR points at the export directory
(default /home/user/data_local/tcrzem/export) and $TCRZEM_CACHE at a cache
directory outside the repository (default /home/user/data_local/tcrzem/cache).
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("TCRZEM_DIR", "/home/user/data_local/tcrzem/export"))
CACHE_DIR = Path(os.environ.get("TCRZEM_CACHE", "/home/user/data_local/tcrzem/cache"))
FS_IN, SEG_POINTS, ONSET_POINT = 512, 1792, 512
EPOCH = os.environ.get("TCRZEM_EPOCH", "standard")
PRE_MS = 200
N_SAMPLES = {"standard": 256, "long": 691}[EPOCH]     # -200..+797 ms or -200..+2496 ms
DECIM = 2                                   # 512 Hz -> 256 Hz
_SUFFIX = "" if EPOCH == "standard" else f"_{EPOCH}"
VALENCES = ("Neg", "Neu", "Pos")            # labels 0, 1, 2 (negative, neutral, pleasant)
N_SCALP = 32
_FNAME = re.compile(r"(S\d+)_Mastoid_(Neg|Neu|Pos)_(High|Low)\.dat$")


def _header(vhdr: Path):
    """Return (n_channels, channel names, resolutions, coordinates (r, theta, phi))."""
    names, res, coords, n_ch = {}, {}, {}, None
    section = None
    for line in vhdr.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
            continue
        if line.startswith("NumberOfChannels="):
            n_ch = int(line.split("=", 1)[1])
        elif section == "[Channel Infos]" and re.match(r"Ch\d+=", line):
            k, v = line.split("=", 1)
            parts = v.split(",")
            names[int(k[2:])] = parts[0]
            res[int(k[2:])] = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
        elif section == "[Coordinates]" and re.match(r"Ch\d+=", line):
            k, v = line.split("=", 1)
            coords[int(k[2:])] = tuple(float(x) for x in v.split(","))
    idx = sorted(names)
    return (n_ch, [names[i] for i in idx], np.array([res[i] for i in idx]),
            np.array([coords.get(i, (0.0, 0.0, 0.0)) for i in idx]))


def montage(data_dir: Path = DATA_DIR):
    """Scalp channel names and BrainVision spherical coordinates (theta, phi in degrees)."""
    vhdr = sorted(data_dir.glob("*_Mastoid_*.vhdr"))[0]
    _, names, _, coords = _header(vhdr)
    return names[:N_SCALP], coords[:N_SCALP, 1:]


def _read_trials(dat: Path):
    """Single trials (n_seg, 256, 32) in microvolts: scalp channels, -200..+800 ms, every 2nd point."""
    n_ch, names, res, _ = _header(dat.with_suffix(".vhdr"))
    assert names[:N_SCALP][0] == "Fp1" and names[N_SCALP - 1] == "Cz", names[:N_SCALP]
    raw = np.fromfile(dat, dtype="<i2")
    n_pts = raw.size // n_ch
    data = raw[: n_pts * n_ch].reshape(n_pts, n_ch)[:, :N_SCALP].astype(np.float32)
    data *= res[:N_SCALP].astype(np.float32)
    n_seg = n_pts // SEG_POINTS
    seg = data[: n_seg * SEG_POINTS].reshape(n_seg, SEG_POINTS, N_SCALP)
    a = ONSET_POINT - PRE_MS * FS_IN // 1000 - 1         # 409: keeps onset on an even offset
    a += (ONSET_POINT - a) % DECIM                        # 410 -> onset at decimated index 51
    b = a + DECIM * N_SAMPLES
    assert b <= SEG_POINTS
    ep = seg[:, a:b, :]
    base = ep[:, : ONSET_POINT - a, :].mean(axis=1, keepdims=True)   # -200..0 ms baseline
    ep = ep - base
    return ep[:, ::DECIM, :]                              # (n_seg, N_SAMPLES, 32), onset at 51


def onset_sample() -> int:
    a = ONSET_POINT - PRE_MS * FS_IN // 1000 - 1
    a += (ONSET_POINT - a) % DECIM
    return (ONSET_POINT - a) // DECIM


def build_cache(data_dir: Path = DATA_DIR, cache_dir: Path = CACHE_DIR, verbose=True):
    """Read every complete-panel subject once; store single trials + index (outside the repo)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in glob.glob(str(data_dir / "*_Mastoid_*.dat")):
        m = _FNAME.search(os.path.basename(f))
        if m:
            files.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(Path(f))
    subjects = sorted(s for s, v in files.items() if all(val in v for val in VALENCES))
    trials, t_subj, t_val, t_arousal = [], [], [], []
    for i, s in enumerate(subjects):
        for vi, val in enumerate(VALENCES):
            for f in sorted(files[s][val]):
                tr = _read_trials(f)
                trials.append(tr)
                t_subj += [s] * len(tr)
                t_val += [vi] * len(tr)
                t_arousal += [1 if "_High" in f.name else 0] * len(tr)
        if verbose and (i + 1) % 25 == 0:
            print(f"[tcrzem] read {i + 1}/{len(subjects)} subjects", flush=True)
    T = np.concatenate(trials).astype(np.float32)
    np.save(cache_dir / f"trials{_SUFFIX}.npy", T)
    np.savez(cache_dir / f"trial_index{_SUFFIX}.npz", subject=np.array(t_subj), valence=np.array(t_val),
             arousal=np.array(t_arousal))
    return T, np.array(t_subj), np.array(t_val), np.array(t_arousal)


def load_trials(cache_dir: Path = CACHE_DIR, mmap=True):
    if not (cache_dir / f"trials{_SUFFIX}.npy").exists():
        return build_cache(cache_dir=cache_dir)
    T = np.load(cache_dir / f"trials{_SUFFIX}.npy", mmap_mode="r" if mmap else None)
    ix = np.load(cache_dir / f"trial_index{_SUFFIX}.npz")
    return T, ix["subject"], ix["valence"], ix["arousal"]


def zscore_epochs(X):
    """Per-observation, per-channel z-score over the full epoch (the SHAPE convention)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def average(T, t_subj, t_val, subjects, n_trials=None, rng=None, zscore=True):
    """Trial-averaged observations X (n_subj*3, N_SAMPLES, 32), y, groups.

    n_trials: if given, each observation averages a random subset of that many trials
    (all trials when fewer are available); rng: numpy Generator for the subset draw.
    """
    X, y, g = [], [], []
    order = {}
    for i, (s, v) in enumerate(zip(t_subj, t_val)):
        order.setdefault((s, int(v)), []).append(i)
    for s in subjects:
        for v in range(len(VALENCES)):
            idx = np.array(order[(s, v)])
            if n_trials is not None and len(idx) > n_trials:
                idx = np.sort(rng.choice(idx, n_trials, replace=False))
            X.append(np.asarray(T[idx], dtype=np.float64).mean(axis=0))
            y.append(v)
            g.append(s)
    X = np.stack(X)
    return (zscore_epochs(X) if zscore else X), np.array(y), np.array(g)


def load_cohort(zscore=True):
    """Observations X (684, N_SAMPLES, 32), y (684,), groups (684,)."""
    T, t_subj, t_val, _ = load_trials()
    subjects = np.unique(t_subj)
    return average(T, t_subj, t_val, subjects, zscore=zscore)


def trial_counts():
    _, t_subj, t_val, _ = load_trials()
    keys, counts = np.unique(np.char.add(t_subj.astype(str), t_val.astype(str)), return_counts=True)
    return counts


if __name__ == "__main__":
    X, y, g = load_cohort()
    c = trial_counts()
    print(f"[tcrzem] X {X.shape}, subjects {np.unique(g).size}, classes {np.bincount(y).tolist()}, "
          f"onset sample {onset_sample()}, trials per observation median {np.median(c):.0f} "
          f"(min {c.min()}, max {c.max()})")
