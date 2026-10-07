#!/usr/bin/env python3
"""J1: conference protocol on the external TCRZEM cohort (clean metrics, controls, fills).

Repeats, on an independent IAPS affective-picture ERP cohort, every fixed-encoder
analysis behind Tables I-II and Fig. 3 of the conference paper (E2 and E5 there),
with the conference code paths (jcore.py is a bit-exact port):

1. Clean three-class metrics per fixed encoder: BA with subject-level 95% CI, macro-F1,
   macro one-vs-rest ROC-AUC from pooled out-of-fold probabilities.
2. Within-subject label-permutation null (20 permutations).
3. Reservoir controls: train-only PCA with K in {16, 32, 64, 128}; pooled (transductive)
   PCA-64 sensitivity check; single-bin code BSC1 vs BSC6 (clean and 30% signal-level
   removal, paired).
4. Centeredness: mean |mu_i|/sigma_i over features on the training subjects of every fold
   (the quantity in Proposition 1), also for the raw BSC6 spike counts before PCA.
5. Feature-coordinate experiment: 10-50% channel dropout x {zero, mean, kNN, spatial}
   fills for every fixed encoder; paired fill-minus-zero contrasts at every level;
   ERP-window minus reservoir at 30% under every fill; raw and above-chance retention.

Protocol: StratifiedGroupKFold(5, shuffle) x seeds 42-46 with subjects as groups,
train-only standardization, balanced L2 logistic readout, OOF probabilities averaged
over partitions, subject-level bootstrap (n_boot 4000). Reservoir rho = 0.9, PCA-64
fitted on the training subjects of each fold. Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import tcrzem_data as TD  # noqa: E402

DROP_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
FILLS = ["zero", "mean", "knn", "spatial"]
K_GRID = [16, 32, 64, 128]
N_PERM = 20
CHANCE = 1.0 / 3.0


def main():
    t0 = time.time()
    X, y, g = TD.load_cohort()
    N, T, n_ch = X.shape
    classes = np.unique(y)
    F = J.folds(y, g)
    band = J.bandpower(X)                     # (N, ch, 5)
    erp = J.erp_windows(X)                    # (N, ch, 3)
    B, rate = J.reservoir_codes(X)            # (N, ch, 1536)
    D = B.shape[2]
    print(f"[J1] N={N} ch={n_ch} subjects={np.unique(g).size} rate={rate:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # train-only PCA embeddings for every K (label-free); keep K=64 per fold with v0
    emb = {k: [] for k in K_GRID}
    for seed, fold, tr, te in F:
        for k in K_GRID:
            pca = PCA(n_components=k, random_state=J.W_SEED).fit(B[tr].reshape(-1, D))
            E = pca.transform(B.reshape(-1, D)).reshape(N, n_ch, k).astype(np.float32)
            v0 = pca.transform(np.zeros((1, D)))[0]
            emb[k].append((E, v0))
    print(f"[J1] embeddings ready ({time.time() - t0:.0f}s)", flush=True)
    blocks = {"Band-power": [band] * len(F), "ERP-window": [erp] * len(F),
              "Reservoir": [e for e, _ in emb[64]]}

    span = "-200..+800 ms" if TD.EPOCH == "standard" else "-200..+2496 ms (pre-specified long epoch)"
    res = {"dataset": "TCRZEM (doi:10.34894/TCRZEM), 228 subjects x 3 valences, 32 scalp channels, "
                      f"{span} at 256 Hz, onset sample 51, per-epoch per-channel z-score",
           "protocol": "StratifiedGroupKFold(5,shuffle) x seeds 42-46; train-only StandardScaler; "
                       "balanced L2 logreg; reservoir rho=0.9, train-only PCA-64; OOF proba averaged "
                       "over partitions; subject-level bootstrap n_boot=%d" % J.N_BOOT,
           "n_obs": int(N), "n_subjects": int(np.unique(g).size), "n_channels": int(n_ch),
           "reservoir_firing_rate": rate}

    # 1. clean metrics
    proba = {n: np.zeros((N, classes.size)) for n in blocks}
    for i, (seed, fold, tr, te) in enumerate(F):
        for n, bl in blocks.items():
            Z = bl[i].reshape(N, -1)
            proba[n][te] += J.fit_pred(Z[tr], y[tr], Z[te], seed)
    clean_pred = {n: J.argmax_pred(p, classes) for n, p in proba.items()}
    res["clean_metrics"] = {n: J.clean_metrics(y, g, proba[n]) for n in blocks}
    res["clean_paired"] = {
        "ERP-window_minus_Reservoir": J.paired_ci(y, g, clean_pred["ERP-window"], clean_pred["Reservoir"]),
        "Band-power_minus_Reservoir": J.paired_ci(y, g, clean_pred["Band-power"], clean_pred["Reservoir"]),
        "ERP-window_minus_Band-power": J.paired_ci(y, g, clean_pred["ERP-window"], clean_pred["Band-power"])}
    for n, r in res["clean_metrics"].items():
        print(f"[J1 clean] {n}: BA={r['BA']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
              f"F1={r['macro_F1']:.3f} AUC={r['macro_OvR_AUC']:.3f}", flush=True)

    # 3a. PCA components
    res["pca_components"] = {}
    for k in K_GRID:
        acc = np.zeros((N, classes.size))
        for i, (seed, fold, tr, te) in enumerate(F):
            Z = emb[k][i][0].reshape(N, -1)
            acc[te] += J.fit_pred(Z[tr], y[tr], Z[te], seed)
        res["pca_components"][str(k)] = J.ba_entry(y, g, J.argmax_pred(acc, classes))
        print(f"[J1 K={k}] {res['pca_components'][str(k)]}", flush=True)
    # 3b. pooled (transductive) PCA sensitivity
    pca_all = PCA(n_components=64, random_state=J.W_SEED).fit(B.reshape(-1, D))
    Eall = pca_all.transform(B.reshape(-1, D)).reshape(N, -1)
    acc = np.zeros((N, classes.size))
    for seed, fold, tr, te in F:
        acc[te] += J.fit_pred(Eall[tr], y[tr], Eall[te], seed)
    pooled_pred = J.argmax_pred(acc, classes)
    res["pooled_pca_sensitivity"] = J.ba_entry(y, g, pooled_pred)
    res["pooled_pca_sensitivity"]["pooled_minus_trainonly"] = J.paired_ci(
        y, g, pooled_pred, clean_pred["Reservoir"])
    print(f"[J1 pooled PCA] {res['pooled_pca_sensitivity']}", flush=True)
    # 3c. temporal binning BSC6 vs BSC1 (clean and 30% signal-level removal)
    B1 = B.reshape(N, n_ch, J.N_RES, J.N_BINS).sum(axis=3)
    p6 = {c: np.zeros((N, classes.size)) for c in ("clean", "signal30")}
    p1 = {c: np.zeros((N, classes.size)) for c in ("clean", "signal30")}
    for i, (seed, fold, tr, te) in enumerate(F):
        drop = J.removal_draw(n_ch, 0.3, seed, fold)
        for acc_, Bc in ((p6, B), (p1, B1)):
            Dc = Bc.shape[2]
            pca = PCA(n_components=64, random_state=J.W_SEED).fit(Bc[tr].reshape(-1, Dc))
            Etr = pca.transform(Bc[tr].reshape(-1, Dc)).reshape(len(tr), -1)
            Ete = pca.transform(Bc[te].reshape(-1, Dc)).reshape(len(te), n_ch, 64)
            acc_["clean"][te] += J.fit_pred(Etr, y[tr], Ete.reshape(len(te), -1), seed)
            Ete[:, drop, :] = pca.transform(np.zeros((1, Dc)))[0]
            acc_["signal30"][te] += J.fit_pred(Etr, y[tr], Ete.reshape(len(te), -1), seed)
    res["temporal_binning"] = {}
    for c in ("clean", "signal30"):
        a6, a1 = J.argmax_pred(p6[c], classes), J.argmax_pred(p1[c], classes)
        pt, ci, _ = J.boot_ci(y, g, {"BSC6": a6, "BSC1": a1})
        res["temporal_binning"][c] = {k: {"BA": pt[k], "ci95": ci[k]} for k in pt}
        res["temporal_binning"][c]["BSC6_minus_BSC1"] = J.paired_ci(y, g, a6, a1)
        print(f"[J1 binning {c}] {res['temporal_binning'][c]}", flush=True)

    # 4. centeredness
    res["centeredness_mean_abs_mu_over_sigma"] = {}
    for n, bl in list(blocks.items()) + [("Reservoir raw BSC6 counts (before PCA)", [B] * len(F))]:
        vals = []
        for i, (seed, fold, tr, te) in enumerate(F):
            Z = bl[i][tr].reshape(len(tr), -1).astype(np.float64)
            sd = Z.std(0)
            ok = sd > 1e-12
            vals.append(float(np.mean(np.abs(Z.mean(0)[ok]) / sd[ok])))
        res["centeredness_mean_abs_mu_over_sigma"][n] = {"mean": float(np.mean(vals)),
                                                         "sd": float(np.std(vals, ddof=1))}
        print(f"[J1 centeredness] {n}: {np.mean(vals):.3f}", flush=True)

    # 5. feature-coordinate fills
    preds = {}
    for frac in DROP_LEVELS:
        for fill in FILLS:
            acc = {n: np.zeros((N, classes.size)) for n in blocks}
            for i, (seed, fold, tr, te) in enumerate(F):
                drop = J.removal_draw(n_ch, frac, seed, fold)
                for n, bl in blocks.items():
                    Z = bl[i].astype(np.float64)
                    Zte = J.fill_block(Z[te], Z[tr], drop, fill)
                    acc[n][te] += J.fit_pred(Z[tr].reshape(len(tr), -1), y[tr],
                                             Zte.reshape(len(te), -1), seed)
            for n in blocks:
                preds[(n, frac, fill)] = J.argmax_pred(acc[n], classes)
        print(f"[J1 fills {int(frac * 100)}%] done ({time.time() - t0:.0f}s)", flush=True)
    names = list(blocks)
    res["dropout"] = {}
    for frac in DROP_LEVELS:
        key = str(int(frac * 100))
        res["dropout"][key] = {}
        for fill in FILLS:
            pm = {n: preds[(n, frac, fill)] for n in names}
            pt, ci, _ = J.boot_ci(y, g, pm)
            ent = {}
            for n in names:
                c = res["clean_metrics"][n]["BA"]
                ent[n] = {"BA": pt[n], "ci95": ci[n], "raw_retention": pt[n] / c,
                          "above_chance_retention": (pt[n] - CHANCE) / (c - CHANCE),
                          "clean_minus_drop": J.paired_ci(y, g, clean_pred[n], pm[n])}
            ent["ERP-window_minus_Reservoir"] = J.paired_ci(y, g, pm["ERP-window"], pm["Reservoir"])
            res["dropout"][key][fill] = ent
    res["fill_minus_zero"] = {n: {fill: {str(int(f * 100)): J.paired_ci(
        y, g, preds[(n, f, fill)], preds[(n, f, "zero")]) for f in DROP_LEVELS}
        for fill in FILLS if fill != "zero"} for n in names}
    for fill in FILLS:
        r = res["dropout"]["30"][fill]
        print(f"[J1 30% {fill}] " + " ".join(f"{n}={r[n]['BA']:.3f}" for n in names)
              + f" ERP-Res={r['ERP-window_minus_Reservoir']['mean_diff']:+.3f} "
              f"[{r['ERP-window_minus_Reservoir']['ci95'][0]:+.3f},{r['ERP-window_minus_Reservoir']['ci95'][1]:+.3f}]",
              flush=True)
    for n in names:
        r = res["fill_minus_zero"][n]["mean"]["30"]
        print(f"[J1 mean-zero 30%] {n}: {r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")

    # 2. permutation null (last: slowest)
    rng = np.random.default_rng(2026)
    null = {n: [] for n in names}
    for p in range(N_PERM):
        yp = y.copy()
        for s in np.unique(g):
            idx = np.where(g == s)[0]
            yp[idx] = rng.permutation(y[idx])
        for n, bl in blocks.items():
            acc = np.zeros((N, classes.size))
            for i, (seed, fold, tr, te) in enumerate(F):
                Z = bl[i].reshape(N, -1)
                acc[te] += J.fit_pred(Z[tr], yp[tr], Z[te], seed)
            null[n].append(float(balanced_accuracy_score(yp, J.argmax_pred(acc, classes))))
        print(f"[J1 perm {p + 1}/{N_PERM}] " + " ".join(f"{n}={null[n][-1]:.3f}" for n in names), flush=True)
    res["permutation_null"] = {n: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                   "max": float(np.max(v)), "n_perm": N_PERM} for n, v in null.items()}
    J.dump(res, "j1_tcrzem_core.json")
    print(f"[J1] total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
