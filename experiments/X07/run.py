"""X07-2026-11: the SPL centering result under leak-free scaling.

Reproduces dissoAdventureExperiments/chapter5Experiments/experiment_zero.py
(the source of the rejected SPL letter's 70.5% -> 88.4% raw-EEG result) and
then changes only the per-epoch scaling.

Replicated exactly from experiment_zero.py: all subjects with the three
condition files (no exclusion), scipy.signal.decimate by floor(1229/256)=4 per
channel, truncation to 256 samples, z-score per channel over those 256 samples
(`epoch_z`); raw EEG flattened (256 x 34); reservoir with per-channel seeds
42 + 17*ch, the same LIF update order, BSC6 over all 256 steps, PCA-64 per
channel; subject centering by each subject's mean over its observations;
10-fold StratifiedGroupKFold (shuffle, seed 42), StandardScaler,
LogisticRegression(C=0.1, max_iter=3000); fold-mean balanced accuracy.

Changes, one at a time:
  scaling : epoch_z (original) | pre_sd (divide by SD of the first 51 decimated
            samples, the pre-stimulus interval) | none
  PCA     : 'all' (original: fitted on every observation) | 'fold'
            (fitted on training subjects of each fold)

Anchor: experiment_zero reports raw EEG 70.5% uncentered, 88.4% centered.
Prediction, fixed before this run: under pre_sd the uncentered and centered
accuracies of both encoders fall, because the condition-dependent offset that
per-epoch z-scoring writes into the pre-stimulus samples survives centering.

Usage: SHAPE_DATA_DIR=... python experiments/X07/run.py
Output: outputs/X07/x07_results.json
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

OUT = ROOT / "outputs" / "X07" / "x07_results.json"
PRE = 51


def load_decimated():
    pat = re.compile(r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$")
    files = {}
    for f in sorted((cm.data_dir() / "general").iterdir()):
        m = pat.search(f.name)
        if m:
            files[(int(m.group(1)), m.group(2))] = f
    subs = sorted({s for s, _ in files})
    X, y, g = [], [], []
    for s in subs:
        for k, c in enumerate(["Neg", "Neu", "Pos"]):
            if (s, c) not in files:
                continue
            raw = np.loadtxt(files[(s, c)])
            factor = max(1, raw.shape[0] // 256)
            target = raw.shape[0] // factor
            d = np.stack([decimate(raw[:, ch], factor)[:target] for ch in range(raw.shape[1])], 1)
            X.append(d[:256]); y.append(k); g.append(s)
    return np.stack(X), np.array(y), np.array(g)


def scale(X, mode):
    if mode == "epoch_z":
        sd = X.std(1, keepdims=True)
        return np.where(sd > 0, (X - X.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1), X)
    if mode == "pre_sd":
        sd = X[:, :PRE].std(1, keepdims=True)   # same zero-SD guard as the original z-score
        return np.where(sd > 0, X / np.where(sd > 0, sd, 1), X)
    if mode == "none":
        return X.copy()
    raise ValueError(mode)


def reservoir_bsc6(X):
    """experiment_zero LIFReservoir per channel (seed 42+17*ch), vectorized over observations."""
    N, T, C = X.shape
    out = np.zeros((N, C, 6 * 256), dtype=np.float32)
    for ch in range(C):
        rng = np.random.RandomState(42 + ch * 17)
        lim_in = np.sqrt(6.0 / 257); W_in = rng.uniform(-lim_in, lim_in, (256, 1))[:, 0]
        lim_rec = np.sqrt(6.0 / 512); W = rng.uniform(-lim_rec, lim_rec, (256, 256))
        W *= 0.9 / np.abs(np.linalg.eigvals(W)).max()
        mem = np.zeros((N, 256)); spk = np.zeros((N, 256)); sp = np.zeros((N, T, 256), dtype=np.float32)
        for t in range(T):
            I = W_in[None, :] * X[:, t, ch][:, None] + spk @ W.T
            mem = 0.95 * mem * (1 - spk) + I
            s = (mem >= 0.5).astype(float)
            mem = np.maximum(mem - s * 0.5, 0)
            sp[:, t] = s; spk = s
        bs = T // 6
        out[:, ch] = np.concatenate([sp[:, b * bs:min((b + 1) * bs, T)].sum(1) for b in range(6)], 1)
    return out


def center(F, g):
    F = F.copy()
    for s in np.unique(g):
        F[g == s] -= F[g == s].mean(0)
    return F


def pca_all(B):
    return np.concatenate([PCA(64).fit_transform(B[:, ch]) for ch in range(B.shape[1])], 1)


def evaluate(F, y, g, B=None, centered=False):
    """experiment_zero.evaluate; with B given, PCA-64 per channel is fitted on the training fold."""
    accs, pred = [], np.zeros(len(y), int)
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g):
        if B is not None:
            Ftr, Fte = [], []
            for ch in range(B.shape[1]):
                p = PCA(64).fit(B[tr, ch]); Ftr.append(p.transform(B[tr, ch])); Fte.append(p.transform(B[te, ch]))
            Ftr, Fte = np.concatenate(Ftr, 1), np.concatenate(Fte, 1)
            if centered:
                Ftr, Fte = center(Ftr, g[tr]), center(Fte, g[te])
        else:
            Ftr, Fte = F[tr], F[te]
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(C=0.1, max_iter=3000, random_state=42).fit(sc.transform(Ftr), y[tr])
        p = clf.predict(sc.transform(Fte)); pred[te] = p
        accs.append(balanced_accuracy_score(y[te], p))
    return {"fold_mean": float(np.mean(accs)), "fold_sd": float(np.std(accs)),
            "pooled_ba": float(balanced_accuracy_score(y, pred))}, pred


def main():
    t0 = time.time()
    X0, y, g = load_decimated()
    print(f"loaded {X0.shape}, subjects {len(np.unique(g))}", flush=True)
    res, preds = {}, {}
    for mode in ["epoch_z", "pre_sd", "none"]:
        X = scale(X0, mode)
        raw = X.reshape(len(X), -1)
        for c in (False, True):
            k = f"raw|{mode}|{'centered' if c else 'uncentered'}"
            res[k], preds[k] = evaluate(center(raw, g) if c else raw, y, g)
            print(k, res[k], f"{time.time() - t0:.0f}s", flush=True)
        B = reservoir_bsc6(X)
        E = pca_all(B)
        for c in (False, True):
            tag = "centered" if c else "uncentered"
            k = f"reservoir_pca_all|{mode}|{tag}"
            res[k], preds[k] = evaluate(center(E, g) if c else E, y, g)
            print(k, res[k], f"{time.time() - t0:.0f}s", flush=True)
            k = f"reservoir_pca_fold|{mode}|{tag}"
            res[k], preds[k] = evaluate(None, y, g, B=B, centered=c)
            print(k, res[k], f"{time.time() - t0:.0f}s", flush=True)
    boot, Bm, names = cm.subject_bootstrap(y, g, preds)
    contrasts = {}
    for enc in ["raw", "reservoir_pca_all", "reservoir_pca_fold"]:
        for c in ["uncentered", "centered"]:
            for m in ["pre_sd", "none"]:
                contrasts[f"{enc}|{c}: epoch_z - {m}"] = cm.paired(Bm, names, f"{enc}|epoch_z|{c}", f"{enc}|{m}|{c}")
            contrasts[f"{enc}|centering gain under epoch_z"] = cm.paired(Bm, names, f"{enc}|epoch_z|centered", f"{enc}|epoch_z|uncentered")
            contrasts[f"{enc}|centering gain under pre_sd"] = cm.paired(Bm, names, f"{enc}|pre_sd|centered", f"{enc}|pre_sd|uncentered")
    cm.write_json(OUT, {"provenance": cm.provenance("X07-2026-11", {
        "source": "dissoAdventureExperiments/chapter5Experiments/experiment_zero.py",
        "n_obs": int(len(y)), "n_subjects": int(len(np.unique(g))), "pre_samples": PRE}),
        "results": res, "pooled_ba_bootstrap": boot, "contrasts": contrasts,
        "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
