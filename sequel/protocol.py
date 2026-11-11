"""Subject-grouped validation shared by every experiment (BIBM protocol, unchanged).

StratifiedGroupKFold(5, shuffle) x seeds 42-46 with subjects as groups; train-only
StandardScaler; balanced L2 logistic readout (C = 1); out-of-fold class probabilities summed
per observation across the five partitions, then argmax; 95% CIs from resampling subjects
with replacement (each keeps all its observations).

The bootstrap is computed from per-subject confusion matrices: a resample is a vector of
subject multiplicities w, its confusion matrix is sum_s w_s N_s, and balanced accuracy follows.
This equals concatenating the resampled subjects' observations, and is fast enough for
4000 resamples of many contrasts.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from . import config as C

Split = tuple[int, int, np.ndarray, np.ndarray]


def make_splits(y: np.ndarray, groups: np.ndarray, seeds: Iterable[int] = C.CV_SEEDS,
                n_folds: int = C.N_FOLDS) -> list[Split]:
    out = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((len(y), 1)), y, groups=groups)):
            out.append((seed, fold, tr, te))
    return out


def fit_readout(Xtr: np.ndarray, ytr: np.ndarray, seed: int, c: float = C.READOUT_C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=c, class_weight="balanced", solver="lbfgs",
                             random_state=seed).fit(sc.transform(Xtr), ytr)
    return sc, clf


FoldFn = Callable[[int, int, np.ndarray, np.ndarray], dict]
Hook = Callable[[str, int, int, StandardScaler, LogisticRegression, dict], None]


def pooled_oof(fold_fn: FoldFn, y: np.ndarray, splits: list[Split],
               hook: Hook | None = None, c: float = C.READOUT_C) -> dict[str, np.ndarray]:
    """Run ``fold_fn`` on every split and pool test probabilities per condition.

    ``fold_fn(seed, fold, tr, te)`` returns ``{condition: (Xtr, Xte, meta)}``. Conditions that
    return the same ``Xtr`` object share one fitted readout. ``hook`` (optional) receives
    ``(condition, seed, fold, scaler, clf, meta)`` for diagnostics. Returns
    ``{condition: predicted labels}``.
    """
    classes = np.unique(y)
    acc: dict[str, np.ndarray] = {}
    for seed, fold, tr, te in splits:
        conds = fold_fn(seed, fold, tr, te)
        fitted: dict[int, tuple] = {}
        for name, (Xtr, Xte, meta) in conds.items():
            key = id(Xtr)
            if key not in fitted:
                fitted[key] = fit_readout(Xtr, y[tr], seed, c)
            sc, clf = fitted[key]
            acc.setdefault(name, np.zeros((len(y), classes.size)))[te] += clf.predict_proba(sc.transform(Xte))
            if hook is not None:
                hook(name, seed, fold, sc, clf, meta)
    return {k: classes[np.argmax(v, axis=1)] for k, v in acc.items()}


# ----------------------------------------------------------------------------- bootstrap
def _subject_confusions(y, groups, pred, classes):
    sids, inv = np.unique(groups, return_inverse=True)
    K = classes.size
    yi = np.searchsorted(classes, y)
    pi = np.searchsorted(classes, pred)
    N = np.zeros((sids.size, K, K))
    np.add.at(N, (inv, yi, pi), 1.0)
    return N


def _ba_from_conf(conf):                    # conf (..., K, K)
    diag = np.diagonal(conf, axis1=-2, axis2=-1)
    rows = conf.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(rows > 0, diag / rows, np.nan)
    return np.nanmean(rec, axis=-1)


def resample_weights(groups: np.ndarray, n_boot: int = C.N_BOOT, seed: int = 7) -> np.ndarray:
    S = np.unique(groups).size
    rng = np.random.default_rng(seed)
    return rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)


def bootstrap_ba(y, groups, preds: dict[str, np.ndarray], n_boot: int = C.N_BOOT,
                 seed: int = 7) -> dict[str, dict]:
    """Point BA and subject-level percentile 95% CI for each prediction vector."""
    classes = np.unique(y)
    Wt = resample_weights(groups, n_boot, seed)
    out = {}
    for name, p in preds.items():
        N = _subject_confusions(y, groups, p, classes)
        boot = _ba_from_conf(np.einsum("bs,skl->bkl", Wt, N))
        out[name] = {"BA": float(balanced_accuracy_score(y, p)),
                     "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def paired_diff(y, groups, pa: np.ndarray, pb: np.ndarray, n_boot: int = C.N_BOOT,
                seed: int = 7) -> dict:
    """BA(pa) - BA(pb) with a paired subject-level bootstrap CI and P(diff > 0)."""
    classes = np.unique(y)
    Wt = resample_weights(groups, n_boot, seed)
    ba = []
    for p in (pa, pb):
        N = _subject_confusions(y, groups, p, classes)
        ba.append(_ba_from_conf(np.einsum("bs,skl->bkl", Wt, N)))
    d = ba[0] - ba[1]
    point = balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb)
    return {"mean_diff": float(point),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt0": float((d > 0).mean())}


# --------------------------------------------------------------------------- permutation
def permute_within_subject(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yp = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        yp[idx] = y[rng.permutation(idx)]
    return yp


def _perm_ba(fold_cache, yp, c):
    classes = np.unique(yp)
    acc = np.zeros((len(yp), classes.size))
    for seed, fold, tr, te, Xtr, Xte in fold_cache:
        sc, clf = fit_readout(Xtr, yp[tr], seed, c)
        acc[te] += clf.predict_proba(sc.transform(Xte))
    return balanced_accuracy_score(yp, classes[np.argmax(acc, axis=1)])


def permutation_test(fold_cache: list, y: np.ndarray, groups: np.ndarray, n_perm: int = 1000,
                     seed: int = 0, n_jobs: int = 1, c: float = C.READOUT_C) -> dict:
    """Within-subject label permutation with the headline estimator.

    ``fold_cache`` holds ``(seed, fold, tr, te, Xtr, Xte)``. Features must be label-free
    (true for every encoder here, including fold-local PCA), and a within-subject
    permutation keeps each subject's label multiset, so the grouped splits are unchanged
    and the cached features stay valid under the null.
    """
    from joblib import Parallel, delayed

    observed = _perm_ba(fold_cache, y, c)
    rng = np.random.default_rng(seed)
    perms = [permute_within_subject(y, groups, rng) for _ in range(n_perm)]
    null = Parallel(n_jobs=n_jobs)(delayed(_perm_ba)(fold_cache, yp, c) for yp in perms)
    null = np.asarray(null)
    return {"observed_BA": float(observed), "n_perm": int(n_perm),
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_max": float(null.max()),
            "p": float((1 + (null >= observed).sum()) / (1 + n_perm))}
