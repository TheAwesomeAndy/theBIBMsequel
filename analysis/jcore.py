#!/usr/bin/env python3
"""Shared core for the journal experiments (dataset-agnostic port of the conference code).

Everything here reproduces the conference implementation in `experiment_camera_ready.py`,
`reanalysis_subject_bootstrap.py`, `experiment2_rawsignal.py` and `perturbations.py`,
generalized so that the channel count, reservoir draw, operating point and analysis
window are parameters instead of constants:

- `reservoir_codes` is `experiment_camera_ready.bsc6` with the weight seed, spectral
  radius, size, leak, threshold, analysis window and bin count as arguments; with the
  defaults it is the conference reservoir bit for bit.
- `boot_ci` / `paired_ci` are vectorized versions of `subject_bootstrap` /
  `paired_diff_ci` that draw the identical subject resamples (same generator calls in
  the same order), so they return the same intervals, only faster.
- `bandpower` is the conference Welch estimator applied along the time axis in one
  call (single 256-point Hann segment, constant detrend, trapezoid over each band).

Protocol (unchanged): StratifiedGroupKFold(5, shuffle) x seeds 42-46, subjects as
groups, train-only StandardScaler, balanced L2 logistic readout, out-of-fold
probabilities averaged per observation over the five partitions, subject-level
bootstrap with every subject's three observations kept together, n_boot = 4000.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 4000
FS = 256
N_RES, BETA, THETA, W_SEED, RHO = 256, 0.05, 0.5, 42, 0.9
T_START, T_END, N_BINS, PCA_K = 10, 70, 6, 64
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
# sample-index milliseconds from the first sample (-200 ms): -120..0, 50..250, 250..600 ms re onset
ERP_WINDOWS_MS = [(80, 200), (250, 450), (450, 800)]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "journal"
# Pre-specified secondary analysis ($TCRZEM_EPOCH=long, see tcrzem_data.py): the epoch runs
# to +2496 ms; the reservoir BSC6 window covers 0..+2484 ms after onset (sample 51, six bins
# of 106 samples) and the ERP-window encoder adds the dataset's two published effect windows
# (500..1300 and 1500..2500 ms after onset). Output files get the suffix "_long".
#
# $JDATA selects the cohort: "tcrzem" (default, public external cohort) or "shape" (the
# conference cohort, restricted, local only; see shape_data.py). On SHAPE, $SHAPE_EPOCH
# selects the analysis windows (pre-specified in memory entry #43, none tuned):
#   standard: the conference input and windows (reservoir t in [10, 70), about -160..+73 ms;
#             ERP windows -120..0 / 50..250 / 250..600 ms after onset), reproduces the
#             conference numbers;
#   onset:    windows measured from true onset (sample 51): reservoir [61, 121) =
#             +39..+273 ms, the window the conference text described, and ERP windows
#             80..200 / 250..450 / 450..800 ms after onset;
#   post:     reservoir [51, 255) = 0..+797 ms (six bins of 34), ERP windows as in onset.
# Band-power always uses the whole epoch. Output files get the cohort name in place of
# "tcrzem" and the epoch suffix.
DATASET = os.environ.get("JDATA", "tcrzem")
if DATASET == "shape":
    EPOCH = os.environ.get("SHAPE_EPOCH", "standard")
    if EPOCH in ("onset", "post"):
        T_START, T_END = (61, 121) if EPOCH == "onset" else (51, 51 + 6 * 34)
        ERP_WINDOWS_MS = [(280, 400), (450, 650), (650, 1000)]
    elif EPOCH != "standard":
        raise ValueError(f"SHAPE_EPOCH={EPOCH}")
else:
    EPOCH = os.environ.get("TCRZEM_EPOCH", "standard")
    if EPOCH == "long":
        T_START, T_END = 51, 51 + 6 * 106
        ERP_WINDOWS_MS = ERP_WINDOWS_MS + [(700, 1500), (1700, 2700)]
SUFFIX = "" if EPOCH == "standard" else f"_{EPOCH}"


def load_cohort(zscore=True):
    """(X, y, g) of the selected cohort: X (N, T, ch), per-epoch z-scored unless zscore=False."""
    if DATASET == "shape":
        import shape_data as SD
        return SD.load_cohort(zscore)
    import tcrzem_data as TD
    return TD.load_cohort(zscore)


def zscore_epochs(X):
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


# ----------------------------------------------------------------------------- folds / readout
def folds(y, g, seeds=SEEDS):
    out = []
    for seed in seeds:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((len(y), 1)), y, groups=g)):
            out.append((seed, fold, tr, te))
    return out


def fit_pred(Xtr, ytr, Xte, seed, return_model=False):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="lbfgs", random_state=seed).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))
    return (p, sc, clf) if return_model else p


def argmax_pred(acc, classes):
    return classes[np.argmax(acc, axis=1)]


# ----------------------------------------------------------------------------- bootstrap
def _resample_weights(g, n_boot, seed):
    """Subject multiplicity matrix (n_boot, n_subj) for the conference resampling stream."""
    rng = np.random.default_rng(seed)
    sid = np.unique(g)
    W = np.zeros((n_boot, sid.size), np.float64)
    for b in range(n_boot):
        pick = rng.choice(sid, size=sid.size, replace=True)
        W[b] = np.bincount(np.searchsorted(sid, pick), minlength=sid.size)
    return sid, W


_WCACHE: dict = {}


def _weights(g, n_boot, seed):
    key = (tuple(np.unique(g).tolist()), n_boot, seed)
    if key not in _WCACHE:
        _WCACHE[key] = _resample_weights(g, n_boot, seed)
    return _WCACHE[key]


def _per_subject_counts(y, g, pred, sid, classes):
    """(n_subj, n_class) correct counts and totals."""
    si = np.searchsorted(sid, g)
    ci = np.searchsorted(classes, y)
    corr = np.zeros((sid.size, classes.size)); tot = np.zeros((sid.size, classes.size))
    np.add.at(tot, (si, ci), 1.0)
    np.add.at(corr, (si, ci), (pred == y).astype(float))
    return corr, tot


def _ba_from_counts(W, corr, tot):
    return ((W @ corr) / (W @ tot)).mean(axis=1)


def ba(y, pred):
    classes = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in classes]))


def boot_ci(y, g, pred_map, n_boot=N_BOOT, seed=12345):
    """Same resamples as reanalysis_subject_bootstrap.subject_bootstrap."""
    sid, W = _weights(g, n_boot, seed)
    classes = np.unique(y)
    pt, ci, boots = {}, {}, {}
    for n, p in pred_map.items():
        corr, tot = _per_subject_counts(y, g, p, sid, classes)
        b = _ba_from_counts(W, corr, tot)
        pt[n] = ba(y, p)
        ci[n] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        boots[n] = b
    return pt, ci, boots


def paired_ci(y, g, predA, predB, n_boot=N_BOOT, seed=999):
    """Same resamples as reanalysis_subject_bootstrap.paired_diff_ci: mean, lo, hi, P(A>B)."""
    sid, W = _weights(g, n_boot, seed)
    classes = np.unique(y)
    a = _ba_from_counts(W, *_per_subject_counts(y, g, predA, sid, classes))
    b = _ba_from_counts(W, *_per_subject_counts(y, g, predB, sid, classes))
    d = a - b
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)),
                                                    float(np.percentile(d, 97.5))],
            "P(A>B)": float(np.mean(d > 0))}


def paired_ci_mean(y, g, pairs, n_boot=N_BOOT, seed=999):
    """Mean over reservoir draws of paired BA differences A - B. Every draw is evaluated on
    the same subject resamples as paired_ci, so the interval is for the draw mean.
    pairs = [(predA, predB), ...], one pair per draw."""
    sid, W = _weights(g, n_boot, seed)
    classes = np.unique(y)
    d = np.mean([_ba_from_counts(W, *_per_subject_counts(y, g, a, sid, classes))
                 - _ba_from_counts(W, *_per_subject_counts(y, g, b, sid, classes)) for a, b in pairs], axis=0)
    point = float(np.mean([ba(y, a) - ba(y, b) for a, b in pairs]))
    return {"point": point, "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "P(A>B)": float(np.mean(d > 0))}


def ba_entry(y, g, pred):
    pt, ci, _ = boot_ci(y, g, {"x": pred})
    return {"BA": pt["x"], "ci95": ci["x"]}


def clean_metrics(y, g, proba):
    classes = np.unique(y)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = argmax_pred(proba, classes)
    e = ba_entry(y, g, pred)
    e["macro_F1"] = float(f1_score(y, pred, average="macro"))
    e["macro_OvR_AUC"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    return e


# ----------------------------------------------------------------------------- encoders
def bandpower(X):
    """X (N, T, ch) -> (N, ch, 5): conference Welch estimator, vectorized over N and ch."""
    T = X.shape[1]
    f, p = welch(X, fs=FS, nperseg=min(T, 256), axis=1)          # (N, F, ch)
    out = np.zeros((X.shape[0], X.shape[2], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f <= hi)
        out[:, :, bi] = np.trapezoid(p[:, m, :], f[m], axis=1)
    return out


def erp_windows(X, windows_ms=ERP_WINDOWS_MS):
    out = np.zeros((X.shape[0], X.shape[2], len(windows_ms)))
    for w, (a, b) in enumerate(windows_ms):
        out[:, :, w] = X[:, int(round(a / 1000 * FS)):int(round(b / 1000 * FS)), :].mean(axis=1)
    return out


def reservoir_weights(rho=RHO, seed=W_SEED, n_res=N_RES):
    """Conference initialization: Xavier-uniform draw, W rescaled to spectral radius rho."""
    rng = np.random.RandomState(seed)
    lin = np.sqrt(6.0 / (1 + n_res)); W_in = rng.uniform(-lin, lin, (n_res, 1))
    lrec = np.sqrt(6.0 / (2 * n_res)); W = rng.uniform(-lrec, lrec, (n_res, n_res))
    if rho == 0:
        W *= 0.0
    else:
        W *= rho / np.abs(np.linalg.eigvals(W)).max()
    return W_in, W


def reservoir_codes(X, rho=RHO, seed=W_SEED, n_res=N_RES, beta=BETA, theta=THETA,
                    t_start=T_START, t_end=T_END, n_bins=N_BINS, chunk=None):
    """Binned spike counts for every (obs, channel) drive.

    Returns codes (N, ch, n_res * n_bins) with column order neuron * n_bins + bin (as in
    the conference code) and the mean firing rate over the analysis window.
    """
    N, T, n_ch = X.shape
    S_all = np.transpose(X, (1, 0, 2)).reshape(T, N * n_ch)
    W_in, W = reservoir_weights(rho, seed, n_res); win0 = W_in[:, 0][:, None]
    binw = (t_end - t_start) // n_bins
    C_all = S_all.shape[1]
    chunk = chunk or C_all
    out = np.zeros((C_all, n_res * n_bins), dtype=np.float32)
    total = 0.0
    for c0 in range(0, C_all, chunk):
        S = S_all[:, c0:c0 + chunk]; C = S.shape[1]
        mem = np.zeros((n_res, C)); spk = np.zeros((n_res, C))
        counts = np.zeros((n_res, n_bins, C), dtype=np.float32)
        for t in range(min(T, t_end)):
            mem = (1.0 - beta) * mem * (1.0 - spk) + win0 * S[t][None, :] + W @ spk
            spk = (mem >= theta).astype(np.float64)
            mem = np.maximum(mem - spk * theta, 0.0)
            if t_start <= t < t_start + binw * n_bins:
                counts[:, (t - t_start) // binw, :] += spk
        total += float(counts.sum())
        out[c0:c0 + C] = counts.reshape(n_res * n_bins, C).T
    rate = total / (n_res * C_all * binw * n_bins)
    return out.reshape(N, n_ch, -1), rate


def damage(X, rho=RHO, seed=W_SEED, n_res=N_RES, beta=BETA, theta=THETA, t_start=T_START,
           t_end=T_END, n_drive=400, rng_seed=0):
    """Damage-spreading order parameter (conference definition): mean normalized Hamming
    distance between two copies after one spike flip at t_start, averaged over the steps
    more than ten after the flip until t_end and over n_drive sampled drives."""
    N, T, n_ch = X.shape
    rng = np.random.default_rng(rng_seed)
    cols = rng.choice(N * n_ch, n_drive, replace=False)
    S = np.transpose(X, (1, 0, 2)).reshape(T, N * n_ch)[:, cols]
    flip = rng.integers(0, n_res, n_drive)
    W_in, W = reservoir_weights(rho, seed, n_res); win0 = W_in[:, 0][:, None]
    mem = [np.zeros((n_res, n_drive)) for _ in range(2)]
    spk = [np.zeros((n_res, n_drive)) for _ in range(2)]
    dist = []
    for t in range(t_end):
        for k in range(2):
            mem[k] = (1.0 - beta) * mem[k] * (1.0 - spk[k]) + win0 * S[t][None, :] + W @ spk[k]
            spk[k] = (mem[k] >= theta).astype(np.float64)
            mem[k] = np.maximum(mem[k] - spk[k] * theta, 0.0)
        if t == t_start:
            spk[1][flip, np.arange(n_drive)] = 1.0 - spk[1][flip, np.arange(n_drive)]
        if t > t_start + 10:
            dist.append(np.abs(spk[0] - spk[1]).mean(axis=0))
    return float(np.mean(dist))


def rho_star(rhos, dmg):
    """Half-maximum crossing of the damage curve, linear interpolation between grid points."""
    rhos = np.asarray(rhos, float); dmg = np.asarray(dmg, float)
    half = 0.5 * dmg.max()
    i = int(np.argmax(dmg >= half))
    if i == 0:
        return float(rhos[0])
    r0, r1, d0, d1 = rhos[i - 1], rhos[i], dmg[i - 1], dmg[i]
    return float(r0 + (half - d0) / (d1 - d0) * (r1 - r0))


def pca_embed(B, tr, k=PCA_K, seed=W_SEED):
    """Train-only PCA shared across electrodes: returns pca, E (N, ch, k), v0 (silent channel)."""
    N, n_ch, D = B.shape
    pca = PCA(n_components=k, random_state=seed).fit(B[tr].reshape(-1, D))
    E = pca.transform(B.reshape(-1, D)).reshape(N, n_ch, k)
    v0 = pca.transform(np.zeros((1, D)))[0]
    return pca, E, v0


# ----------------------------------------------------------------------------- fills
def removal_draw(n_ch, frac, seed, fold):
    """Conference draw (perturbations.removal_draw / run_oof): default_rng(1000*seed + fold)."""
    if frac <= 0:
        return np.array([], int)
    rng = np.random.default_rng(1000 * seed + fold)
    return rng.choice(n_ch, int(round(frac * n_ch)), replace=False)


def fill_block(Xte_blk, Xtr_blk, drop, fill, knn_k=10):
    """Feature-coordinate fill on (n, ch, f) blocks; returns a filled copy (raw feature space)."""
    out = Xte_blk.copy()
    if drop.size == 0:
        return out
    if fill == "zero":
        out[:, drop, :] = 0.0
    elif fill == "mean":
        out[:, drop, :] = Xtr_blk[:, drop, :].mean(axis=0)
    elif fill == "spatial":
        keep = np.setdiff1d(np.arange(out.shape[1]), drop)
        out[:, drop, :] = out[:, keep, :].mean(axis=1, keepdims=True)
    elif fill == "knn":
        from sklearn.impute import KNNImputer
        n, c, f = out.shape
        flat = out.reshape(n, -1).copy()
        cols = (drop[:, None] * f + np.arange(f)[None, :]).ravel()
        flat[:, cols] = np.nan
        imp = KNNImputer(n_neighbors=knn_k).fit(Xtr_blk.reshape(len(Xtr_blk), -1))
        out = imp.transform(flat).reshape(n, c, f)
    else:
        raise ValueError(fill)
    return out


# ----------------------------------------------------------------------------- io
def dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if DATASET != "tcrzem":
        name = name.replace("_tcrzem_", f"_{DATASET}_")
    if SUFFIX and not name.endswith(SUFFIX + ".json"):
        name = name.replace(".json", SUFFIX + ".json")
    p = OUT / name
    json.dump(obj, open(p, "w"), indent=2)
    print(f"[out] wrote {p.relative_to(OUT.parents[2])}", flush=True)
    return p
