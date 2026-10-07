#!/usr/bin/env python3
"""Subject-level paired-bootstrap re-analysis of the channel-missingness
feature-coordinate audit.

Reproduces the fixed-encoder comparison (band-power, ERP-window, reservoir
BSC6/PCA embedding) under four fill rules (zero, train-mean, kNN, spatial)
across channel-dropout levels, using the source protocol:
StratifiedGroupKFold(5, shuffle) x seeds 42-46, train-only StandardScaler,
balanced L2 logistic readout. It captures per-observation out-of-fold
predictions, aggregates them per subject across the five repeated partitions,
and computes subject-level bootstrap 95% CIs for:

  * clean and channel-missingness balanced accuracy (BA),
  * paired absolute BA change (clean - dropped), and
  * paired differences between encoders and between fill rules,

resampling the 211 subjects with replacement and preserving all three
condition observations per subject (the correct resampling unit; fold means
are NOT treated as independent).

Privacy: reads the restricted SHAPE feature pickle locally (path from
$ARSPI_SHAPE_FEATURES or the sibling operational-distinctness-paper/data dir).
It never writes subject-level data; only aggregate de-identified JSON is saved.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import balanced_accuracy_score

SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
DROP_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
FILLS = ["zero", "mean", "knn", "spatial"]
N_BOOT = 4000
FS = 256  # Hz
# ERP windows (ms) -> sample indices on the 256-sample post-stimulus epoch.
# Windows are in sample-index milliseconds (sample / 256 Hz), measured from the FIRST sample of
# X_ds, which is 200 ms before stimulus onset (X_ds == raw[0:1024:4]; onset at raw sample 205,
# X_ds sample 51). Relative to onset they span about -120..0, 50..250 and 250..600 ms.
ERP_WINDOWS_MS = [(80, 200), (250, 450), (450, 800)]


def _resolve_pickle() -> Path:
    env = os.environ.get("ARSPI_SHAPE_FEATURES")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base.parent / "operational-distinctness-paper" / "data" / "shape_features_211.pkl"
        if cand.exists():
            return cand
    cand = Path("/home/user/operational-distinctness-paper/data/shape_features_211.pkl")
    if cand.exists():
        return cand
    raise SystemExit("SHAPE feature pickle not found; set $ARSPI_SHAPE_FEATURES")


def build_blocks(d):
    """Return {name: (X (N,D), ch_shape (n_ch,n_feat))} for the three fixed encoders."""
    N = d["y"].shape[0]
    band = d["conv_feats"].astype(float)           # (N, 34, 5)
    res = d["lsm_bsc6_pca"].astype(float)          # (N, 34, 64)
    if np.allclose(res.std(), 0.0):
        # pickle ships the reservoir block zeroed; use the faithfully
        # regenerated embedding (analysis/regen_reservoir_embedding.py; validated
        # against published clean BA 0.463). Never committed (subject-level).
        npy = os.environ.get("ARSPI_RESERVOIR_NPY", "/tmp/reservoir_bsc6_pca_211.npy")
        res = np.load(npy).astype(float)
        print(f"[blocks] reservoir block was zeroed; loaded regenerated {res.shape} from {npy}")
    Xds = d["X_ds"].astype(float)                  # (N, 256, 34)
    # ERP-window: mean amplitude in three post-stimulus windows, per channel.
    n_ch = Xds.shape[2]
    erp = np.zeros((N, n_ch, len(ERP_WINDOWS_MS)))
    for w, (a_ms, b_ms) in enumerate(ERP_WINDOWS_MS):
        a = int(round(a_ms / 1000.0 * FS))
        b = int(round(b_ms / 1000.0 * FS))
        erp[:, :, w] = Xds[:, a:b, :].mean(axis=1)
    return {
        "Band-power": (band.reshape(N, -1), band.shape[1:]),
        "ERP-window": (erp.reshape(N, -1), erp.shape[1:]),
        "Reservoir":  (res.reshape(N, -1), res.shape[1:]),
    }


def drop_channels(ch_shape, frac, rng):
    n_ch, n_feat = ch_shape
    n_drop = int(round(frac * n_ch))
    drop = rng.choice(n_ch, size=n_drop, replace=False) if n_drop else np.array([], int)
    col_mask = np.ones((n_ch, n_feat), bool)
    if n_drop:
        col_mask[drop, :] = False
    return drop, col_mask.reshape(-1), n_feat


def fill_test(Xte, Xtr, ch_shape, drop_ch, col_keep, fill):
    """Return a copy of Xte with dropped-channel columns filled by `fill` (raw space)."""
    if drop_ch.size == 0:
        return Xte
    n_ch, n_feat = ch_shape
    out = Xte.copy()
    dropped_cols = np.where(~col_keep)[0]
    if fill == "zero":
        out[:, dropped_cols] = 0.0
    elif fill == "mean":
        out[:, dropped_cols] = Xtr[:, dropped_cols].mean(axis=0, keepdims=True)
    elif fill == "knn":
        Xte_nan = out.copy()
        Xte_nan[:, dropped_cols] = np.nan
        imp = KNNImputer(n_neighbors=10)
        imp.fit(Xtr)
        out = imp.transform(Xte_nan)
    elif fill == "spatial":
        # per observation, per feature index: mean over that obs's retained channels
        keep_ch = np.array([c for c in range(n_ch) if c not in set(drop_ch.tolist())])
        M = out.reshape(out.shape[0], n_ch, n_feat)
        retained_mean = M[:, keep_ch, :].mean(axis=1)          # (n_obs, n_feat)
        for c in drop_ch:
            M[:, c, :] = retained_mean
        out = M.reshape(out.shape[0], -1)
    return out


def run_oof(X, ch_shape, y, groups, frac, fill):
    """Return (N,) aggregated OOF predictions: mean proba across seeds -> argmax."""
    N = X.shape[0]
    classes = np.unique(y)
    proba_sum = np.zeros((N, classes.size))
    seen = np.zeros(N)
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
            Xtr, Xte = X[tr], X[te]
            if frac > 0:
                rng = np.random.default_rng(1000 * seed + fold)
                drop_ch, col_keep, _ = drop_channels(ch_shape, frac, rng)
                Xte = fill_test(Xte, Xtr, ch_shape, drop_ch, col_keep, fill)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     solver="lbfgs", random_state=seed)
            clf.fit(sc.transform(Xtr), y[tr])
            pr = clf.predict_proba(sc.transform(Xte))
            proba_sum[te] += pr
            seen[te] += 1
    agg = proba_sum / np.maximum(seen[:, None], 1)
    return classes[np.argmax(agg, axis=1)]


def subject_bootstrap(y, subjects, pred_map, n_boot=N_BOOT, seed=12345):
    """Bootstrap subjects (with replacement); return callable stats.

    pred_map: dict name -> (N,) aggregated predictions.
    Returns per-name BA CI and pairwise/paired-change CIs over the same resamples.
    """
    rng = np.random.default_rng(seed)
    subj_ids = np.unique(subjects)
    subj_to_idx = {s: np.where(subjects == s)[0] for s in subj_ids}
    names = list(pred_map.keys())

    def ba_for(idx, name):
        return balanced_accuracy_score(y[idx], pred_map[name][idx])

    point = {n: ba_for(np.arange(len(y)), n) for n in names}
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        pick = rng.choice(subj_ids, size=subj_ids.size, replace=True)
        idx = np.concatenate([subj_to_idx[s] for s in pick])
        for n in names:
            boot[n].append(ba_for(idx, n))
    ci = {n: (float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5)))
          for n in names}
    return point, ci, boot


def paired_diff_ci(y, subjects, predA, predB, n_boot=N_BOOT, seed=999):
    rng = np.random.default_rng(seed)
    subj_ids = np.unique(subjects)
    subj_to_idx = {s: np.where(subjects == s)[0] for s in subj_ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(subj_ids, size=subj_ids.size, replace=True)
        idx = np.concatenate([subj_to_idx[s] for s in pick])
        diffs.append(balanced_accuracy_score(y[idx], predA[idx])
                     - balanced_accuracy_score(y[idx], predB[idx]))
    diffs = np.array(diffs)
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float(np.mean(diffs > 0)))


def main():
    pkl = _resolve_pickle()
    d = pickle.load(open(pkl, "rb"))
    y = np.asarray(d["y"]); groups = np.asarray(d["subjects"])
    blocks = build_blocks(d)
    print(f"[data] N={len(y)} subjects={np.unique(groups).size} classes={np.bincount(y).tolist()}")

    # --- clean OOF predictions per encoder ---
    preds = {"clean": {}, }
    for name, (X, ch) in blocks.items():
        preds["clean"][name] = run_oof(X, ch, y, groups, 0.0, "zero")
    pt, ci, _ = subject_bootstrap(y, groups, preds["clean"])
    print("[clean] subject-bootstrap BA (95% CI):")
    for n in blocks:
        print(f"   {n:11s} {pt[n]:.4f}  [{ci[n][0]:.4f}, {ci[n][1]:.4f}]")

    # --- dropout x fill OOF predictions ---
    results = {"protocol": "StratifiedGroupKFold(5,shuffle) x seeds 42-46; "
               "train-only StandardScaler; balanced L2 logreg; "
               "subject-level bootstrap (211 subjects, 3 obs each) n_boot=%d" % N_BOOT,
               "clean_BA": {n: {"point": pt[n], "ci95": ci[n]} for n in blocks}}
    drop_preds = {}  # (name,frac,fill) -> preds
    for frac in DROP_LEVELS:
        for fill in FILLS:
            for name, (X, ch) in blocks.items():
                drop_preds[(name, frac, fill)] = run_oof(X, ch, y, groups, frac, fill)
        print(f"[drop {int(frac*100)}%] fills computed")

    # 30% dropout table with subject-bootstrap CIs + paired change vs clean
    focus = 0.3
    results["dropout_30"] = {}
    for fill in FILLS:
        pm = {n: drop_preds[(n, focus, fill)] for n in blocks}
        p2, c2, _ = subject_bootstrap(y, groups, pm)
        entry = {}
        for n in blocks:
            dmean, dlo, dhi, _ = paired_diff_ci(y, groups, preds["clean"][n], pm[n])
            entry[n] = {"drop_BA": p2[n], "drop_ci95": c2[n],
                        "clean_minus_drop": {"mean": dmean, "ci95": [dlo, dhi]}}
        results["dropout_30"][fill] = entry

    # Key paired encoder comparisons at 30% (ERP-window vs Reservoir) per fill
    results["paired_ERPwindow_minus_Reservoir_30"] = {}
    for fill in FILLS:
        m, lo, hi, pgt = paired_diff_ci(
            y, groups, drop_preds[("ERP-window", focus, fill)],
            drop_preds[("Reservoir", focus, fill)])
        results["paired_ERPwindow_minus_Reservoir_30"][fill] = {
            "mean_diff": m, "ci95": [lo, hi], "P(ERPwindow>Reservoir)": pgt}

    # Full dropout curve (point BA) per fill per encoder for the figure
    results["curve_BA"] = {}
    for name in blocks:
        results["curve_BA"][name] = {}
        for fill in FILLS:
            results["curve_BA"][name][fill] = {
                f"{int(f*100)}": float(balanced_accuracy_score(
                    y, drop_preds[(name, f, fill)])) for f in DROP_LEVELS}

    out = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "subject_bootstrap_reanalysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"[out] wrote {out}")

    # human-readable summary of the headline result
    print("\n=== 30% dropout, subject-bootstrap BA [95% CI] ===")
    for fill in FILLS:
        e = results["dropout_30"][fill]
        print(f" fill={fill:8s} "
              + " | ".join(f"{n[:4]}={e[n]['drop_BA']:.3f}[{e[n]['drop_ci95'][0]:.3f},{e[n]['drop_ci95'][1]:.3f}]"
                           for n in blocks))
    print("\n=== ERP-window minus Reservoir @30% (paired subject-bootstrap) ===")
    for fill in FILLS:
        r = results["paired_ERPwindow_minus_Reservoir_30"][fill]
        print(f" fill={fill:8s} diff={r['mean_diff']:+.4f} "
              f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"P(ERP>Res)={r['P(ERPwindow>Reservoir)']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
