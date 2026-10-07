#!/usr/bin/env python3
"""J7: permutation null for decoding from the pre-stimulus interval alone (external cohort).

ERP-window features computed only from -200..0 ms (one mean amplitude per channel) under
per-epoch z-scoring (Z) and one fixed per-channel scale (G, as in J6), scored under the
conference protocol, against a within-subject label-permutation null with 200 permutations
(each subject keeps one observation per class). Reports the observed BA, the null mean, SD
and 95th/99th percentiles, and the permutation p-value (fraction of null BAs >= observed,
with the +1 correction). Runs on the epoch selected by $TCRZEM_EPOCH. Aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402

N_PERM = 200


def oof(Xf, yfit, F, N, classes):
    acc = np.zeros((N, classes.size))
    for seed, fold, tr, te in F:
        acc[te] += J.fit_pred(Xf[tr], yfit[tr], Xf[te], seed)
    return J.argmax_pred(acc, classes)


def main():
    t0 = time.time()
    Xuv, y, g = J.load_cohort(zscore=False)
    N = len(y); classes = np.unique(y)
    F = J.folds(y, g)
    feats = {"Z": J.erp_windows(J.zscore_epochs(Xuv), [(0, 200)]).reshape(N, -1),
             "G": J.erp_windows(Xuv / Xuv.std(axis=(0, 1)), [(0, 200)]).reshape(N, -1)}
    res = {"epoch": J.EPOCH, "n_perm": N_PERM,
           "protocol": "ERP-window pre-only (-200..0 ms) features; conference protocol; within-subject "
                       "label permutations; p = (1 + #null >= observed) / (1 + n_perm)", "results": {}}
    rng = np.random.default_rng(7)
    perms = []
    for _ in range(N_PERM):
        yp = y.copy()
        for s in np.unique(g):
            idx = np.where(g == s)[0]
            yp[idx] = rng.permutation(y[idx])
        perms.append(yp)
    for nm, Xf in feats.items():
        obs = J.ba(y, oof(Xf, y, F, N, classes))
        null = np.array([balanced_accuracy_score(yp, oof(Xf, yp, F, N, classes)) for yp in perms])
        res["results"][nm] = {"BA": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                              "null_p95": float(np.percentile(null, 95)), "null_p99": float(np.percentile(null, 99)),
                              "null_max": float(null.max()),
                              "p_value": float((1 + np.sum(null >= obs)) / (1 + N_PERM))}
        print(f"[J7 {J.EPOCH} {nm}] BA={obs:.3f} null mean={null.mean():.3f} p95={np.percentile(null, 95):.3f} "
              f"p={res['results'][nm]['p_value']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    J.dump(res, "j7_tcrzem_prestim_null.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
