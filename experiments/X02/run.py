"""X02-2026-11: pre-stimulus decoding after per-trial normalization, five ERP CORE paradigms.

Design follows the statistic pre-registered in TheAwesomeAndy/dissoAdventuresv2
(experiments/zscore_artifact/MECHANISM.md and run_erpcore.py, commit 69b6c97,
committed before any ERP CORE result): time-resolved decoding in 4-sample
(15.6 ms) bins of the 28-channel pattern, test statistic = mean accuracy over
the bins whose centers lie in [-200, 0) ms. This run extends that design from
N170 and P3 to five stimulus-locked paradigms and adds the identity check.
Two earlier, stopped runs with other features are recorded in
outputs/X02/SUPERSEDED_RUNS.md.

Regimes
  within : single trials per participant, shrinkage LDA per bin, stratified
           5-fold x2; group test: one-sample t-test of participant means vs 0.5
  across : trials normalized, then averaged per participant and condition
           (pre-registered order); participant-grouped 5-fold x5, standardized
           L2 logistic per bin; null: 200 permutations swapping each
           participant's two labels with probability 1/2
  across_avgnorm (secondary): average first, then normalize the average, the
           order used for SHAPE, whose delivered epochs are averages
Identity (secondary): across regime, decoder fed -m/sd of each average.

Predictions, fixed before this run: under epoch_z and post_z the pre-stimulus
statistic exceeds its null across participants in paradigms with a sustained
one-signed condition difference, and is at chance under none and pre_sd;
within participants the leak is smaller than across participants.

Run record: the first run computed N170, P3, N400 with one permutation stream
(default_rng(2026)) shared across paradigms and was stopped during N2pc by a
session interruption. N2pc and MMN were then computed with `run.py N2pc MMN`,
which seeds each paradigm's permutations separately. This changes only which
random permutations are drawn.

Usage: ERPCORE_DATA_DIR=... python experiments/X02/run.py [PARADIGM ...]
Output: outputs/X02/x02_results.json
"""
from __future__ import annotations

import json
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
BIN = 4
N_PERM = 200
OUT = ROOT / "outputs" / "X02" / "x02_results.json"


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def binned(x):
    n, T, c = x.shape
    nb = T // BIN
    return x[:, :nb * BIN].reshape(n, nb, BIN, c).mean(2)


def within(xb, y, seed):
    nb = xb.shape[1]; acc = np.zeros(nb)
    for r in range(2):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(xb[:, 0], y):
            for b in range(nb):
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                acc[b] += bal(y[te], clf.fit(xb[tr, b], y[tr]).predict(xb[te, b])) / 10
    return acc


def across(A, y, g, bins, repeats=5, seed=0):
    acc = np.zeros(len(bins))
    for r in range(repeats):
        pred = np.zeros((len(bins), len(y)), int)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(A[:, 0], y, g):
            for j, b in enumerate(bins):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                pred[j, te] = clf.fit(A[tr, b], y[tr]).predict(A[te, b])
        acc += np.array([bal(y, pred[j]) for j in range(len(bins))]) / repeats
    return acc


def flat_across(F, y, g, repeats=5, seed=0):
    return float(across(F[:, None, :], y, g, [0], repeats, seed)[0])


def main(only=None):
    """`only`: paradigms to (re)compute; results already in OUT for other paradigms are kept.
    The permutation stream is seeded per paradigm so a resumed run matches a full run."""
    t0 = time.time()
    res = json.loads(OUT.read_text())["results"] if (only and OUT.exists()) else {}
    for par in ec.CONTRASTS:
        if only and par not in only:
            continue
        rng = np.random.default_rng([2026, list(ec.CONTRASTS).index(par)])
        parts = ec.participants(par)
        W = {n: [] for n in NORMS}; A = {n: [] for n in NORMS}; Aavg = {n: [] for n in NORMS}
        ident, g, y2, counts = [], [], [], []
        centers = None
        for pid, f in parts:
            x, times, y, _ = ec.load(f, par)
            counts.append([int((y == 0).sum()), int((y == 1).sum())])
            centers = times[:(len(times) // BIN) * BIN].reshape(-1, BIN).mean(1)
            avg_raw = np.stack([x[y == c].mean(0) for c in (0, 1)])
            for n in NORMS:
                xb = binned(ec.normalize(x, times, n))
                W[n].append(within(xb, y, seed=pid))
                A[n] += [xb[y == c].mean(0) for c in (0, 1)]
                Aavg[n] += list(binned(ec.normalize(avg_raw, times, n)))
            ident += list(-avg_raw.mean(1) / avg_raw.std(1))
            g += [pid, pid]; y2 += [0, 1]
        g = np.array(g); y2 = np.array(y2)
        pre = np.flatnonzero((centers >= -200) & (centers < 0))
        allb = np.arange(len(centers))
        out = {"n_participants": len(parts), "trials_per_class_median": np.median(counts, 0).tolist(),
               "min_trials_per_class": int(np.min(counts)), "bin_center_ms": centers.round(2).tolist(),
               "pre_bins": pre.tolist(), "within": {}, "across": {}, "across_avgnorm": {}}
        out["identity_across_ba"] = flat_across(np.array(ident), y2, g)
        for n in NORMS:
            Wn = np.array(W[n]); pv = Wn[:, pre].mean(1)
            t, p = stats.ttest_1samp(pv, 0.5)
            out["within"][n] = {"pre_mean": float(pv.mean()),
                                "ci": [float(pv.mean() - 1.96 * stats.sem(pv)), float(pv.mean() + 1.96 * stats.sem(pv))],
                                "t": float(t), "p": float(p), "n_above_half": int((pv > 0.5).sum()),
                                "time_course": Wn.mean(0).round(4).tolist(),
                                "time_course_sem": stats.sem(Wn, axis=0).round(4).tolist()}
            for key, src in (("across", A[n]), ("across_avgnorm", Aavg[n])):
                Arr = np.stack(src)
                tc = across(Arr, y2, g, allb)
                obs = float(tc[pre].mean())
                null = []
                for _ in range(N_PERM):
                    flip = rng.random(len(parts)) < 0.5
                    yp = y2.reshape(-1, 2).copy(); yp[flip] = yp[flip][:, ::-1]
                    null.append(across(Arr, yp.ravel(), g, pre, repeats=1, seed=int(rng.integers(1e6))).mean())
                null = np.array(null)
                out[key][n] = {"pre_mean": obs, "p": float((1 + np.sum(null >= obs)) / (1 + N_PERM)),
                               "null_mean": float(null.mean()), "null_q95": float(np.percentile(null, 95)),
                               "time_course": tc.round(4).tolist()}
            print(f"{par:5s} {n:8s} within {pv.mean():.3f} p={p:.2g} | across {out['across'][n]['pre_mean']:.3f} "
                  f"p={out['across'][n]['p']:.3f} | avg-then-norm {out['across_avgnorm'][n]['pre_mean']:.3f} "
                  f"p={out['across_avgnorm'][n]['p']:.3f}  {time.time() - t0:.0f}s", flush=True)
        print(f"{par:5s} identity -m/sd across {out['identity_across_ba']:.3f}", flush=True)
        res[par] = out
        cm.write_json(OUT, {"provenance": {"experiment": "X02-2026-11", "norms": NORMS, "bin_samples": BIN,
                                           "n_perm": N_PERM, "pre_window_ms": [-200, 0],
                                           "preregistered_design": "dissoAdventuresv2@69b6c97"},
                            "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main(sys.argv[1:] or None)
