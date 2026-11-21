"""X02-2026-11: pre-stimulus decoding after per-trial normalization, five ERP CORE paradigms.

Predictions, fixed in this file before any X02 result was computed:
  P1 (across participants: one average per participant and condition)
     pre-stimulus decoding is above its permutation null under epoch_z and
     post_z and within it under none and pre_sd, in every paradigm whose
     condition difference has a sustained post-stimulus mean (P3, N400, MMN,
     N170). N2pc is lateralized and largely sign-balanced across hemispheres,
     so its leak can be smaller.
  P2 (identity) under epoch_z, pre-stimulus decoding matches a decoder fed
     -m/sd, the epoch mean over the epoch SD, to within sampling error.
  P3 (within participants, single trials) the leak is smaller than across
     participants (the averaging law, tested directly in X03).

Revision (2026-11-21, before any X03 run and after the first X02 run on N170
and P3): the pre-stimulus mean feature was replaced by four pre-stimulus bin
means (sequel.erpcore.pre_bins). The mean is identically zero after baseline
correction, up to float32 rounding, and the first run decoded that residue
(P3, unscaled input, BA 0.643). The identity feature -m/sd is unchanged.

Usage: ERPCORE_DATA_DIR=... python experiments/X02/run.py
Output: outputs/X02/x02_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402
from sequel import erpcore as ec  # noqa: E402

NORMS = ["none", "pre_sd", "epoch_z", "post_z"]
N_PERM = 1000
OUT = ROOT / "outputs" / "X02" / "x02_results.json"


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def across_cv(F, y, g, repeats=5, seed=0):
    accs = []
    for r in range(repeats):
        pred = np.zeros(len(y), int)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(F, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            pred[te] = clf.fit(F[tr], y[tr]).predict(F[te])
        accs.append(bal(y, pred))
    return float(np.mean(accs))


def within_cv(F, y, seed=0):
    acc = 0.0
    for r in range(2):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(F, y):
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            acc += bal(y[te], clf.fit(F[tr], y[tr]).predict(F[te])) / 10
    return acc


def main():
    t0 = time.time()
    res = {}
    for par in ec.CONTRASTS:
        parts = ec.participants(par)
        if not parts:
            print(par, "no data"); continue
        avgs = {n: [] for n in NORMS}; ids, labs = [], []
        within = {n: [] for n in NORMS}; within_id = []
        n_trials = []
        for pid, f in parts:
            x, times, y, _ = ec.load(f, par)
            pre = times < 0
            n_trials.append(len(y))
            avg_raw = np.stack([x[y == c].mean(0) for c in (0, 1)])
            for n in NORMS:
                # across regime: normalize the participant average, as SHAPE does
                A = ec.normalize(avg_raw, times, n)
                avgs[n].append(ec.pre_bins(A, times))
                # within regime: normalize each trial
                Z = ec.normalize(x, times, n)
                within[n].append(within_cv(ec.pre_bins(Z, times), y))
            avgs.setdefault("identity_-m/sd", []).append(-avg_raw.mean(1) / avg_raw.std(1))
            ids += [pid, pid]; labs += [0, 1]
            within_id.append(pid)
        g = np.array(ids); y2 = np.array(labs)
        out = {"n_participants": len(parts), "trials_median": float(np.median(n_trials)),
               "across": {}, "within": {}}
        rng = np.random.default_rng(2026)
        Fi = np.stack(avgs["identity_-m/sd"]).reshape(len(parts) * 2, -1)
        out["identity_-m/sd_across_ba"] = across_cv(Fi, y2, g)
        for n in NORMS:
            F = np.stack(avgs[n]).reshape(len(parts) * 2, -1)
            obs = across_cv(F, y2, g)
            null = []
            for _ in range(N_PERM):
                flip = rng.random(len(parts)) < 0.5
                yp = y2.copy().reshape(-1, 2); yp[flip] = yp[flip][:, ::-1]
                null.append(across_cv(F, yp.ravel(), g, repeats=1, seed=int(rng.integers(1e6))))
            null = np.array(null)
            out["across"][n] = {"ba": obs, "null_q95": float(np.percentile(null, 95)),
                                "null_mean": float(null.mean()),
                                "p": float((1 + np.sum(null >= obs)) / (1 + N_PERM))}
            w = np.array(within[n])
            t, p = stats.ttest_1samp(w, 0.5)
            lo, hi = stats.t.interval(0.95, len(w) - 1, loc=w.mean(), scale=stats.sem(w))
            out["within"][n] = {"ba_mean": float(w.mean()), "ci": [float(lo), float(hi)],
                                "t": float(t), "p": float(p)}
            print(f"{par:5s} {n:8s} across {obs:.3f} (p={out['across'][n]['p']:.3f})  "
                  f"within {w.mean():.3f} [{lo:.3f},{hi:.3f}] p={p:.3g}  {time.time()-t0:.0f}s", flush=True)
        res[par] = out
    cm.write_json(OUT, {"provenance": {"experiment": "X02-2026-11", "norms": NORMS, "n_perm": N_PERM,
                                       "feature": "four 12-sample pre-stimulus bin means ending at onset (-188..0 ms), 28 scalp channels",
                                       "across": "participant-grouped 5-fold x5, StandardScaler + L2 logistic",
                                       "within": "stratified 5-fold x2, shrinkage LDA"},
                        "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
