"""Shared components for the BIBM sequel experiments.

Data loading, the normalization variants under study, the BIBM reservoir
encoder (imported arithmetic, see `reservoir_bsc`), and the subject-grouped
evaluation protocol of the accepted BIBM paper.

Restricted data never enter this repository. The SHAPE directory is read from
the environment variable SHAPE_DATA_DIR, which must contain `general/` with the
per-subject trial-averaged ERP text files (1229 samples x 34 channels, 1024 Hz,
baseline-corrected over the first 205 samples).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# --- BIBM constants (identical to the accepted analysis, analysis/core.py) --
N_RES, BETA, THETA, W_SEED = 256, 0.05, 0.5, 42
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
N_BOOT = 4000
FS_DS = 256          # analysed rate after taking every 4th sample
ONSET_DS = 51        # stimulus onset, analysed-sample index
ONSET_RAW = 205      # stimulus onset, raw 1024 Hz sample index
EXCLUDE = {127}      # recording anomaly, excluded in BIBM and the dissertation
CONDITIONS = ["IAPSNeg", "IAPSNeu", "IAPSPos"]


# --- data -----------------------------------------------------------------
def data_dir() -> Path:
    d = os.environ.get("SHAPE_DATA_DIR")
    if not d:
        raise SystemExit("Set SHAPE_DATA_DIR to the restricted SHAPE directory.")
    return Path(d)


def load_shape_raw():
    """Return raw (N, 1229, 34) baseline-corrected microvolts, labels y, subject ids g."""
    root = data_dir() / "general"
    pat = re.compile(r"SHAPE_Community_(\d+)_(IAPSNeg|IAPSNeu|IAPSPos)_BC\.txt$")
    rec = {}
    for f in sorted(root.iterdir()):
        m = pat.search(f.name)
        if m and int(m.group(1)) not in EXCLUDE:
            rec[(int(m.group(1)), m.group(2))] = f
    subs = sorted({s for s, _ in rec})
    subs = [s for s in subs if all((s, c) in rec for c in CONDITIONS)]
    X, y, g = [], [], []
    for s in subs:
        for k, c in enumerate(CONDITIONS):
            X.append(np.loadtxt(rec[(s, c)]))
            y.append(k)
            g.append(s)
    return np.stack(X), np.array(y), np.array(g)


def data_checksum() -> str:
    """SHA-256 over the sorted file names and sizes of the general/ directory."""
    h = hashlib.sha256()
    for f in sorted((data_dir() / "general").iterdir()):
        h.update(f"{f.name}:{f.stat().st_size}\n".encode())
    return h.hexdigest()


def downsample(raw):
    """BIBM analysed signal before scaling: raw[:, 0:1024:4, :] (256 samples, -200..+800 ms)."""
    return raw[:, 0:1024:4, :]


# --- normalization variants -----------------------------------------------
def normalize(X, mode: str, pre: int = ONSET_DS):
    """Per-channel scaling of each epoch. X is (N, T, C) baseline-corrected.

    epoch_z  : z-score over the whole epoch (the BIBM and dissertation choice)
    post_z   : z-score with mean and SD from post-stimulus samples only
    pre_sd   : divide by the epoch's own pre-stimulus SD (no post-stimulus sample used)
    none     : baseline-corrected microvolts, unscaled
    global_sd: divide by one cohort-wide SD per channel (label-free, shared by all epochs)
    """
    if mode == "epoch_z":
        return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    if mode == "post_z":
        P = X[:, pre:, :]
        return (X - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
    if mode == "pre_sd":
        return X / X[:, :pre, :].std(1, keepdims=True)
    if mode == "none":
        return X.copy()
    if mode == "global_sd":
        return X / X.std(axis=(0, 1), keepdims=True)
    raise ValueError(mode)


# --- BIBM reservoir ---------------------------------------------------------
def init_weights(rho: float = 0.9, seed: int = W_SEED):
    """Xavier-uniform draw of the accepted analysis; W rescaled to spectral radius rho."""
    rng = np.random.RandomState(seed)
    lin = np.sqrt(6.0 / (1 + N_RES)); W_in = rng.uniform(-lin, lin, (N_RES, 1))
    lrec = np.sqrt(6.0 / (2 * N_RES)); W = rng.uniform(-lrec, lrec, (N_RES, N_RES))
    if rho > 0:
        W *= rho / np.abs(np.linalg.eigvals(W)).max()
    else:
        W[:] = 0.0
    return W_in, W


def reservoir_bsc(X, t_start=10, t_end=70, n_bins=6, rho=0.9, seed=W_SEED):
    """Spike-bin counts for every (obs, channel) drive: (N, C, N_RES * n_bins), mean rate.

    Same LIF update and addition order as analysis/core.lif_step of the accepted
    paper. The loop stops at t_end because later steps never enter the code.
    """
    N, T, C = X.shape
    S = np.transpose(X, (1, 0, 2)).reshape(T, N * C)
    W_in, W = init_weights(rho, seed); win0 = W_in[:, 0][:, None]
    mem = np.zeros((N_RES, N * C)); spk = np.zeros((N_RES, N * C))
    counts = np.zeros((N_RES, n_bins, N * C), dtype=np.float32)
    binw = (t_end - t_start) // n_bins
    for t in range(t_end):
        mem = (1.0 - BETA) * mem * (1.0 - spk) + win0 * S[t][None, :] + W @ spk
        spk = (mem >= THETA).astype(np.float64)
        mem = np.maximum(mem - spk * THETA, 0.0)
        if t >= t_start:
            counts[:, min((t - t_start) // binw, n_bins - 1), :] += spk
    rate = float(counts.sum() / (N_RES * N * C * (t_end - t_start)))
    return counts.reshape(N_RES * n_bins, N * C).T.reshape(N, C, -1), rate


# --- evaluation protocol (BIBM Sec. IV-B) -----------------------------------
def _block_pca(Ftr, Fte, k):
    """One PCA basis shared across electrodes, fitted on training rows only."""
    n_tr, C, D = Ftr.shape
    p = PCA(n_components=k, random_state=42).fit(Ftr.reshape(-1, D))
    return (p.transform(Ftr.reshape(-1, D)).reshape(n_tr, -1),
            p.transform(Fte.reshape(-1, D)).reshape(Fte.shape[0], -1))


def oof_proba(F, y, g, block_pca_k=None):
    """Out-of-fold class probabilities summed over the five seeded partitions.

    F is (N, d) or, with block_pca_k, (N, C, D) blocks compressed fold-locally.
    """
    P = np.zeros((len(y), len(np.unique(y))))
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in cv.split(np.zeros(len(y)), y, g):
            if block_pca_k:
                Xtr, Xte = _block_pca(F[tr], F[te], block_pca_k)
            else:
                Xtr, Xte = F[tr], F[te]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     solver="lbfgs", random_state=seed)
            clf.fit(sc.transform(Xtr), y[tr])
            P[te] += clf.predict_proba(sc.transform(Xte))
    return P


def balanced_accuracy(y, pred):
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def subject_bootstrap(y, g, preds: dict, n_boot=N_BOOT, seed=0):
    """Balanced accuracy per method with 95% subject-level percentile intervals.

    Subjects are resampled with replacement and their observations kept
    together. Returns point estimates, intervals, and the bootstrap matrix for
    paired contrasts.
    """
    subs = np.unique(g); idx = {s: np.flatnonzero(g == s) for s in subs}
    rng = np.random.default_rng(seed)
    names = list(preds)
    B = np.zeros((n_boot, len(names)))
    for b in range(n_boot):
        rows = np.concatenate([idx[s] for s in rng.choice(subs, len(subs))])
        for j, n in enumerate(names):
            B[b, j] = balanced_accuracy(y[rows], preds[n][rows])
    out = {}
    for j, n in enumerate(names):
        lo, hi = np.percentile(B[:, j], [2.5, 97.5])
        out[n] = {"ba": balanced_accuracy(y, preds[n]), "ci": [float(lo), float(hi)]}
    return out, B, names


def paired(B, names, a, b):
    d = B[:, names.index(a)] - B[:, names.index(b)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_mean_boot": float(d.mean()), "ci": [float(lo), float(hi)]}


def label_permutation_null(F, y, g, n_perm=200, seed=2026, block_pca_k=None, n_seeds=1):
    """Within-subject label permutation: each subject's three labels shuffled together."""
    rng = np.random.default_rng(seed)
    subs = np.unique(g); out = []
    global SEEDS
    keep = SEEDS
    try:
        SEEDS = keep[:n_seeds]
        for _ in range(n_perm):
            yp = y.copy()
            for s in subs:
                r = np.flatnonzero(g == s); yp[r] = rng.permutation(y[r])
            out.append(balanced_accuracy(yp, oof_proba(F, yp, g, block_pca_k).argmax(1)))
    finally:
        SEEDS = keep
    return np.array(out)


# --- provenance -------------------------------------------------------------
def provenance(exp_id: str, params: dict) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = "unknown"
    return {
        "experiment": exp_id,
        "git_commit": commit,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "data_checksum": data_checksum(),
        "params": params,
    }


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
