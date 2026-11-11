#!/usr/bin/env python3
"""Camera-ready experiment E5: controls behind the Validation and Results statements.

1. Label-permutation null. Condition labels are permuted WITHIN each subject (each subject
   keeps one observation per class, so the design stays balanced), the full protocol is
   rerun, and balanced accuracy is scored against the permuted labels. Reported per encoder
   as mean, SD and max over N_PERM permutations.
2. PCA component count. Reservoir (rho = 0.9) with train-only PCA at K in {16, 32, 64, 128}
   components per electrode block; subject-level BA [95% CI].
3. Temporal binning. Single-bin code BSC1 (total post-onset spike count per unit; the six
   BSC6 bins summed) through the identical train-only PCA-64 pipeline, clean and at 30%
   signal-level electrode removal, with the paired subject-level difference BSC6 - BSC1.
4. Centeredness (Proposition 1). For each encoder, the mean over features of |mu_i|/sigma_i,
   computed on the training subjects of every fold (the quantity that sets the zero-fill
   shift in Eq. noninv), averaged over the 25 folds.
5. Clean metrics (Table I): balanced accuracy with subject-level 95% CI, macro-F1, and
   macro one-vs-rest ROC-AUC from the pooled out-of-fold probabilities, for every encoder.

Protocol: StratifiedGroupKFold(5) x seeds 42-46, subject groups, train-only standardization,
balanced L2 logistic; OOF probabilities averaged over partitions; subject-level bootstrap.
Band power uses the Welch estimator of experiment2_rawsignal.py. Writes aggregate JSON only.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment2_rawsignal as E2S  # noqa: E402
import experiment_camera_ready as EC  # noqa: E402
import reanalysis_subject_bootstrap as RB  # noqa: E402

N_PERM = 20
K_GRID = [16, 32, 64, 128]


def oof_pred(feats_per_fold, y_fit, folds, N, classes):
    """feats_per_fold: list aligned with folds of (Xtr, Xte). Returns aggregated predictions."""
    acc = np.zeros((N, classes.size))
    for (seed, tr, te), (Xtr, Xte) in zip(folds, feats_per_fold):
        acc[te] += EC.fit_pred(Xtr, y_fit[tr], Xte, seed)
    return classes[np.argmax(acc, axis=1)]


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    N = len(y); classes = np.unique(y); t0 = time.time()
    band = E2S.bandpower_all(X).reshape(N, -1); erp = E2S.erp_all(X).reshape(N, -1)
    B, _ = EC.bsc6(X, 0.9); D = B.shape[2]
    folds = []
    for seed in EC.SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        folds += [(seed, tr, te) for tr, te in cv.split(np.zeros((N, 1)), y, groups=g)]
    # train-only PCA embeddings for every K (PCA is label-free, so the permutation null reuses K = 64)
    emb = {k: [] for k in K_GRID}
    for seed, tr, te in folds:
        for k in K_GRID:
            pca = PCA(n_components=k, random_state=EC.SEED).fit(B[tr].reshape(-1, D))
            emb[k].append((pca.transform(B[tr].reshape(-1, D)).reshape(len(tr), -1),
                           pca.transform(B[te].reshape(-1, D)).reshape(len(te), -1)))
    print(f"[E5] embeddings ready ({time.time() - t0:.0f}s)", flush=True)
    res = {"protocol": "StratifiedGroupKFold(5) x seeds 42-46; train-only StandardScaler; balanced L2 "
           "logreg; train-only PCA; OOF proba averaged over partitions; subject-level bootstrap "
           "n_boot=%d; permutation within subject, N_PERM=%d" % (RB.N_BOOT, N_PERM),
           "pca_components": {}, "permutation_null": {}}
    for k in K_GRID:
        p = oof_pred(emb[k], y, folds, N, classes)
        pt, ci, _ = RB.subject_bootstrap(y, g, {"Reservoir": p})
        res["pca_components"][str(k)] = {"BA": pt["Reservoir"], "ci95": list(ci["Reservoir"])}
        print(f"[E5 K={k}] reservoir BA={pt['Reservoir']:.3f} [{ci['Reservoir'][0]:.3f},{ci['Reservoir'][1]:.3f}]", flush=True)
    # 3. temporal binning: BSC1 = six bins summed (column order is neuron*6 + bin)
    B1 = B.reshape(N, B.shape[1], EC.N_RES, EC.BSC_N_BINS).sum(axis=3)
    p6 = {"clean": np.zeros((N, classes.size)), "signal30": np.zeros((N, classes.size))}
    p1 = {"clean": np.zeros((N, classes.size)), "signal30": np.zeros((N, classes.size))}
    for i, (seed, tr, te) in enumerate(folds):
        fold = i % 5
        drop = np.random.default_rng(1000 * seed + fold).choice(B.shape[1], int(round(0.3 * B.shape[1])), replace=False)
        for code, acc_, Bc in (("bsc6", p6, B), ("bsc1", p1, B1)):
            Dc = Bc.shape[2]
            pca = PCA(n_components=64, random_state=EC.SEED).fit(Bc[tr].reshape(-1, Dc))
            Etr = pca.transform(Bc[tr].reshape(-1, Dc)).reshape(len(tr), -1)
            Ete = pca.transform(Bc[te].reshape(-1, Dc)).reshape(len(te), B.shape[1], 64)
            acc_["clean"][te] += EC.fit_pred(Etr, y[tr], Ete.reshape(len(te), -1), seed)
            Ete[:, drop, :] = pca.transform(np.zeros((1, Dc)))[0]      # removed electrode = silent drive
            acc_["signal30"][te] += EC.fit_pred(Etr, y[tr], Ete.reshape(len(te), -1), seed)
    res["temporal_binning"] = {}
    for cond in ("clean", "signal30"):
        a6 = classes[np.argmax(p6[cond], axis=1)]; a1 = classes[np.argmax(p1[cond], axis=1)]
        pt, ci, _ = RB.subject_bootstrap(y, g, {"BSC6": a6, "BSC1": a1})
        m, lo, hi, _ = RB.paired_diff_ci(y, g, a6, a1)
        res["temporal_binning"][cond] = {k: {"BA": pt[k], "ci95": list(ci[k])} for k in pt}
        res["temporal_binning"][cond]["BSC6_minus_BSC1"] = {"mean_diff": m, "ci95": [lo, hi]}
        print(f"[E5 binning {cond}] BSC6={pt['BSC6']:.3f} BSC1={pt['BSC1']:.3f} diff={m:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    # 4. centeredness of each encoder on the training folds
    res["centeredness_mean_abs_mu_over_sigma"] = {}
    for name, fl in (("Band-power", [(band[tr], None) for _, tr, te in folds]),
                     ("ERP-window", [(erp[tr], None) for _, tr, te in folds]),
                     ("Reservoir", emb[64])):
        vals = [float(np.mean(np.abs(Xtr.mean(0)) / (Xtr.std(0) + 1e-12))) for Xtr, _ in fl]
        res["centeredness_mean_abs_mu_over_sigma"][name] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
        print(f"[E5 centeredness] {name}: mean |mu|/sigma = {np.mean(vals):.3f}", flush=True)
    # 5. clean metrics for Table I: BA [subject CI], macro-F1, macro one-vs-rest AUC (pooled OOF proba)
    res["clean_metrics"] = {}
    for name, fl in (("Band-power", [(band[tr], band[te]) for _, tr, te in folds]),
                     ("ERP-window", [(erp[tr], erp[te]) for _, tr, te in folds]),
                     ("Reservoir", emb[64])):
        proba = np.zeros((N, classes.size))
        for (seed, tr, te), (Xtr, Xte) in zip(folds, fl):
            proba[te] += EC.fit_pred(Xtr, y[tr], Xte, seed)
        proba /= proba.sum(axis=1, keepdims=True)
        pred = classes[np.argmax(proba, axis=1)]
        pt, ci, _ = RB.subject_bootstrap(y, g, {name: pred})
        res["clean_metrics"][name] = {"BA": pt[name], "BA_ci95": list(ci[name]),
                                      "macro_F1": float(f1_score(y, pred, average="macro")),
                                      "macro_OvR_AUC": float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))}
        r = res["clean_metrics"][name]
        print(f"[E5 clean] {name}: BA={r['BA']:.3f} F1={r['macro_F1']:.3f} AUC={r['macro_OvR_AUC']:.3f}", flush=True)
    feats = {"Band-power": [(band[tr], band[te]) for _, tr, te in folds],
             "ERP-window": [(erp[tr], erp[te]) for _, tr, te in folds],
             "Reservoir": emb[64]}
    rng = np.random.default_rng(2026)
    null = {n: [] for n in feats}
    for i in range(N_PERM):
        yp = y.copy()
        for s in np.unique(g):
            idx = np.where(g == s)[0]; yp[idx] = rng.permutation(y[idx])
        for n, f in feats.items():
            null[n].append(balanced_accuracy_score(yp, oof_pred(f, yp, folds, N, classes)))
        print(f"[E5 perm {i + 1}/{N_PERM}] " + " ".join(f"{n}={null[n][-1]:.3f}" for n in feats), flush=True)
    for n, v in null.items():
        res["permutation_null"][n] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                      "max": float(np.max(v))}
    out = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "e5_controls.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[out] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
