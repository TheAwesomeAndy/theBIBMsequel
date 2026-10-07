#!/usr/bin/env python3
"""External check on DEAP (Koelstra et al., 2012): the channel-dropout fill-rule effect on an
independent affective EEG dataset (camera-ready: committed so the published numbers are
reproducible from the repository).

Data: the DEAP "data_preprocessed_python" release (s01.dat ... s32.dat; 40 trials x 40
channels x 8064 samples at 128 Hz; labels = valence, arousal, dominance, liking). DEAP is
distributed under its own end-user licence; obtain it from the dataset authors and point
$DEAP_DIR at the folder containing the s*.dat files. Nothing from DEAP is committed.

Protocol (as published in the paper):
  * the first 3 s (384 samples, pre-trial baseline) are dropped; EEG = first 32 channels;
  * band power per channel: Welch (nperseg 256), mean PSD in 4-8, 8-12, 12-30, 30-45 Hz,
    log-transformed -> 128 features;
  * task: binary valence (rating >= 5);
  * WITHIN-subject evaluation (cross-subject band power is at chance on DEAP): for every
    subject with >= 5 trials per class, StratifiedKFold(5, shuffle) x seeds 42-46;
    train-only StandardScaler + balanced L2 logistic readout;
  * channel dropout of 10/30/50% of electrodes in the test fold, filled by zero,
    train-mean, or kNN (k = 10, fit on the training fold).
Reported: BA averaged over (subject, fold, seed) as in the paper, plus a subject-level
bootstrap 95% CI over the per-subject mean BA (subjects resampled with replacement).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
FS, NCH = 128, 32
BANDS = [(4, 8), (8, 12), (12, 30), (30, 45)]
SEEDS = [42, 43, 44, 45, 46]
RATES = [0.10, 0.30, 0.50]
FILLS = ["zero", "mean", "knn"]
N_BOOT = 4000


def bandpower(trial):
    f, P = welch(trial[:NCH], fs=FS, nperseg=min(256, trial.shape[1]))
    out = np.stack([P[:, (f >= lo) & (f < hi)].mean(axis=1) for lo, hi in BANDS], axis=1)
    return np.log(out + 1e-12)                                   # (32, 4)


def load(deap_dir):
    files = sorted(glob.glob(os.path.join(deap_dir, "**", "s*.dat"), recursive=True))
    if len(files) != 32:
        raise SystemExit(f"expected 32 DEAP subject files under {deap_dir}, found {len(files)}")
    X, y, g = [], [], []
    for si, fn in enumerate(files):
        d = pickle.load(open(fn, "rb"), encoding="latin1")
        data = np.asarray(d["data"])[:, :, 384:]
        lab = (np.asarray(d["labels"])[:, 0] >= 5).astype(int)
        for t in range(data.shape[0]):
            X.append(bandpower(data[t]).reshape(-1)); y.append(lab[t]); g.append(si)
    return np.array(X), np.array(y), np.array(g)


def fill(Xtr, Xte, drop_cols, how):
    Y = Xte.copy()
    if how == "zero":
        Y[:, drop_cols] = 0.0
    elif how == "mean":
        Y[:, drop_cols] = Xtr.mean(axis=0)[drop_cols]
    else:
        Y[:, drop_cols] = np.nan
        Y = KNNImputer(n_neighbors=10).fit(Xtr).transform(Y)
    return Y


def fit_pred(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs",
                             random_state=seed).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def evaluate(X, y, g, frac, how):
    """Per-subject list of fold BAs (over seeds and folds)."""
    per_subject = {}
    for seed in SEEDS:
        for s in np.unique(g):
            idx = np.where(g == s)[0]; ys = y[idx]
            if np.bincount(ys, minlength=2).min() < 5:
                continue
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            for k, (a, b) in enumerate(cv.split(idx, ys)):
                tr, te = idx[a], idx[b]
                Xte = X[te]
                if frac > 0:
                    rng = np.random.default_rng(1000 * seed + 7 * k + int(idx[0]))
                    ch = rng.choice(NCH, int(round(frac * NCH)), replace=False)
                    cols = (ch[:, None] * len(BANDS) + np.arange(len(BANDS))[None, :]).reshape(-1)
                    Xte = fill(X[tr], Xte, cols, how)
                per_subject.setdefault(int(s), []).append(
                    balanced_accuracy_score(y[te], fit_pred(X[tr], y[tr], Xte, seed)))
    return per_subject


def summarize(per_subject, rng_seed=7):
    allv = np.concatenate([np.asarray(v) for v in per_subject.values()])
    subj = np.array([np.mean(v) for v in per_subject.values()])
    rng = np.random.default_rng(rng_seed)
    boot = [rng.choice(subj, subj.size, replace=True).mean() for _ in range(N_BOOT)]
    return {"BA_mean_over_folds": float(allv.mean()), "subject_mean_BA": float(subj.mean()),
            "subject_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_subjects": int(subj.size)}


def main():
    deap_dir = os.environ.get("DEAP_DIR", "/home/user/data_local/deap")
    X, y, g = load(deap_dir)
    print(f"[DEAP] {X.shape[0]} trials, {len(set(g.tolist()))} subjects, classes {np.bincount(y).tolist()}", flush=True)
    res = {"protocol": __doc__.split("Protocol (as published in the paper):")[1].split("Reported:")[0].strip(),
           "BandPower": {"clean": summarize(evaluate(X, y, g, 0.0, "zero"))}}
    print(f"[DEAP clean] {res['BandPower']['clean']['BA_mean_over_folds']:.4f}", flush=True)
    for frac in RATES:
        res["BandPower"][f"{frac:.2f}"] = {}
        for how in FILLS:
            r = summarize(evaluate(X, y, g, frac, how))
            res["BandPower"][f"{frac:.2f}"][how] = r
            print(f"[DEAP {int(frac * 100)}% {how}] {r['BA_mean_over_folds']:.4f} "
                  f"(subject CI {r['subject_ci95'][0]:.3f}-{r['subject_ci95'][1]:.3f})", flush=True)
    out = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "deap_replication_v2.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[out] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
