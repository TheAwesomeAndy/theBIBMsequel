#!/usr/bin/env python3
"""J5: direct test of Proposition 1 (zero-fill depends on the coordinate origin).

Proposition 1: with train-only standardization the clean readout is invariant to a
constant offset x -> x - b, but a zero-filled coordinate enters the readout as
z_i = -(mu_i - b_i)/sigma_i, so the dropout score depends on b. Equivalently, filling
the dropped block with the raw value x_i = mu_i + kappa * sigma_i places it at
standardized position z_i = kappa: kappa = 0 is train-mean fill and the encoder's own
zero sits at z_i = -mu_i/sigma_i.

For every fixed encoder (band-power, ERP-window, reservoir rho = 0.9 with train-only
PCA-64) this script:
1. checks clean invariance: the readout is refitted on features shifted by
   b = mu + 2 sigma (fold-wise training statistics) and the out-of-fold class
   probabilities are compared with the unshifted ones (max absolute difference, which
   is bounded by the L-BFGS stopping tolerance, and the share of predicted labels that
   differ, per fold);
2. scores channel dropout (10/30/50%, conference draws) with the dropped block placed at
   kappa in {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}, and with the encoder's native zero;
3. computes, per test observation at 30% dropout, the logit change between native-zero
   and mean fill predicted by Eq. (noninv), delta_l = W_S (-mu_S / sigma_S) (class-centered
   norm), and the fraction of predictions it flips.
Protocol and inference as in jcore.py. Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import tcrzem_data as TD  # noqa: E402

KAPPAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
LEVELS = [0.1, 0.3, 0.5]
NAMES = ("Band-power", "ERP-window", "Reservoir")


def main():
    t0 = time.time()
    X, y, g = TD.load_cohort()
    N, T, n_ch = X.shape
    classes = np.unique(y)
    band, erp = J.bandpower(X), J.erp_windows(X)
    B, _ = J.reservoir_codes(X)
    F = J.folds(y, g)
    fills = [f"kappa_{k:+.1f}" for k in KAPPAS] + ["native_zero"]
    acc = {(n, f, lv): np.zeros((N, classes.size)) for n in NAMES for f in fills for lv in LEVELS}
    clean = {n: np.zeros((N, classes.size)) for n in NAMES}
    shifted = {n: np.zeros((N, classes.size)) for n in NAMES}
    dlogit = {n: [] for n in NAMES}
    flips = {n: [] for n in NAMES}
    inv_labels = {n: [] for n in NAMES}
    zpos = {n: [] for n in NAMES}
    for seed, fold, tr, te in F:
        _, E, _ = J.pca_embed(B, tr)
        blocks = {"Band-power": band, "ERP-window": erp, "Reservoir": E.astype(np.float64)}
        for n, Z in blocks.items():
            nf = Z.shape[2]
            Ztr, Zte = Z[tr].reshape(len(tr), -1), Z[te].reshape(len(te), -1)
            p, sc, clf = J.fit_pred(Ztr, y[tr], Zte, seed, return_model=True)
            clean[n][te] += p
            mu, sd = sc.mean_, sc.scale_
            b = mu + 2.0 * sd
            ps = J.fit_pred(Ztr - b, y[tr], Zte - b, seed)
            shifted[n][te] += ps
            inv_labels[n].append(float(np.mean(np.argmax(ps, 1) != np.argmax(p, 1))))
            for lv in LEVELS:
                drop = J.removal_draw(n_ch, lv, seed, fold)
                cols = (drop[:, None] * nf + np.arange(nf)[None, :]).ravel()
                for f in fills:
                    Zf = Zte.copy()
                    Zf[:, cols] = 0.0 if f == "native_zero" else mu[cols] + float(f.split("_")[1]) * sd[cols]
                    acc[(n, f, lv)][te] += clf.predict_proba(sc.transform(Zf))
                if lv == 0.3:
                    W = clf.coef_                                   # (classes, features)
                    dl = W[:, cols] @ (-mu[cols] / sd[cols])         # zero minus mean, per class
                    dl = dl - dl.mean()
                    dlogit[n].append(float(np.linalg.norm(dl)))
                    zpos[n].append(float(np.mean(np.abs(mu[cols] / sd[cols]))))
                    Zz = Zte.copy(); Zz[:, cols] = 0.0
                    Zm = Zte.copy(); Zm[:, cols] = mu[cols]
                    flips[n].append(float(np.mean(clf.predict(sc.transform(Zz)) != clf.predict(sc.transform(Zm)))))
        print(f"[J5] seed {seed} fold {fold} ({time.time() - t0:.0f}s)", flush=True)
    res = {"protocol": "fills placed at standardized position kappa (raw value mu + kappa*sigma, training "
                       "statistics) or native zero; conference removal draws; reservoir rho=0.9 with "
                       "train-only PCA-64; StratifiedGroupKFold(5) x seeds 42-46; balanced L2 logreg; OOF "
                       "proba averaged over partitions; subject-level bootstrap n_boot=%d" % J.N_BOOT,
           "kappas": KAPPAS, "clean_invariance_max_abs_proba_diff": {},
           "clean_invariance_fraction_labels_changed_per_fold_max": {n: float(np.max(v)) for n, v in inv_labels.items()},
           "clean_invariance_fraction_labels_changed_mean": {n: float(np.mean(v)) for n, v in inv_labels.items()},
           "clean_BA": {}, "dropout": {},
           "predicted_logit_shift_30": {}}
    for n in NAMES:
        res["clean_invariance_max_abs_proba_diff"][n] = float(np.max(np.abs(clean[n] - shifted[n])))
        cp = J.argmax_pred(clean[n], classes)
        res["clean_BA"][n] = J.ba_entry(y, g, cp)
        res["dropout"][n] = {}
        for lv in LEVELS:
            key = str(int(lv * 100))
            pm = {f: J.argmax_pred(acc[(n, f, lv)], classes) for f in fills}
            pt, ci, _ = J.boot_ci(y, g, pm)
            ent = {f: {"BA": pt[f], "ci95": ci[f]} for f in fills}
            ent["native_minus_mean"] = J.paired_ci(y, g, pm["native_zero"], pm["kappa_+0.0"])
            ent["kappa-2_minus_kappa+2"] = J.paired_ci(y, g, pm["kappa_-2.0"], pm["kappa_+2.0"])
            ent["kappa0_minus"] = {f: J.paired_ci(y, g, pm["kappa_+0.0"], pm[f])
                                   for f in fills if f != "kappa_+0.0"}
            res["dropout"][n][key] = ent
        res["predicted_logit_shift_30"][n] = {
            "mean_class_centered_norm": float(np.mean(dlogit[n])), "sd": float(np.std(dlogit[n], ddof=1)),
            "mean_abs_mu_over_sigma_dropped": float(np.mean(zpos[n])),
            "fraction_predictions_flipped_zero_vs_mean": float(np.mean(flips[n]))}
        e = res["dropout"][n]["30"]
        print(f"[J5 {n}] inv={res['clean_invariance_max_abs_proba_diff'][n]:.2e} clean={res['clean_BA'][n]['BA']:.3f} "
              f"native0={e['native_zero']['BA']:.3f} " + " ".join(f"k{k:+.1f}={e[f'kappa_{k:+.1f}']['BA']:.3f}" for k in KAPPAS)
              + f" |dl|={np.mean(dlogit[n]):.2f} flips={np.mean(flips[n]):.3f}", flush=True)
    J.dump(res, "j5_tcrzem_origin.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
