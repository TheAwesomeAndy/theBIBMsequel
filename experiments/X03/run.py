"""X03-2026-11: the averaging law for normalization leakage (ERP CORE, within participants).

Theory (paper Sec. III-C): under epoch_z the pre-stimulus mean of a
baseline-corrected epoch equals -(T_post/T) * mean_post / sd. For an average of
k independent trials the condition difference in mean_post is unchanged while
its trial noise variance falls as 1/k, so the discriminability of the leaked
feature grows as sqrt(k):  D(k) = sqrt(k) D(1).  For equal-variance Gaussian
classes, BA = Phi(D/2), hence the prediction, fixed before this run:

    BA_pred(k) = Phi( sqrt(k) * Phi^{-1}( BA_obs(1) ) )

with BA_obs(1) the observed single-trial leak. The division by sd and
finite-sample estimation make this an approximation; the test is whether the
observed curve follows it. Under pre_sd the leak is predicted to stay at 0.5
for every k.

Design: per participant, stratified 5-fold split of trials; within each fold,
R pseudo-trials per class are formed from the training pool and from the test
pool separately by averaging k distinct random trials, then normalized,
reduced to the pre-stimulus mean pattern, and decoded with shrinkage LDA.

Usage: ERPCORE_DATA_DIR=... python experiments/X03/run.py
Output: outputs/X03/x03_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402
from sequel import erpcore as ec  # noqa: E402

KS = [1, 2, 4, 8, 16]
NORMS = ["epoch_z", "pre_sd"]
R = 40
OUT = ROOT / "outputs" / "X03" / "x03_results.json"


def pseudo(x, k, R, rng):
    if k == 1:
        return x
    return np.stack([x[rng.choice(len(x), k, replace=False)].mean(0) for _ in range(R)])


def participant_curve(x, times, y, norm, rng):
    pre = times < 0
    out = {}
    for k in KS:
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x[:, 0, 0], y):
            pools = {}
            ok = True
            for split, idx in (("tr", tr), ("te", te)):
                Xs, Ys = [], []
                for c in (0, 1):
                    xc = x[idx][y[idx] == c]
                    if len(xc) < k:
                        ok = False; break
                    P = pseudo(xc, k, R, rng)
                    Xs.append(ec.normalize(P, times, norm)[:, pre].mean(1)); Ys.append(np.full(len(P), c))
                if not ok:
                    break
                pools[split] = (np.concatenate(Xs), np.concatenate(Ys))
            if not ok:
                break
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            clf.fit(*pools["tr"])
            Xt, yt = pools["te"]
            p = clf.predict(Xt)
            accs.append(np.mean([(p[yt == c] == c).mean() for c in (0, 1)]))
        if len(accs) == 5:
            out[k] = float(np.mean(accs))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    res = {}
    for par in ec.CONTRASTS:
        parts = ec.participants(par)
        curves = {n: {k: [] for k in KS} for n in NORMS}
        for pid, f in parts:
            x, times, y, _ = ec.load(f, par)
            for n in NORMS:
                for k, a in participant_curve(x, times, y, n, rng).items():
                    curves[n][k].append(a)
        out = {}
        for n in NORMS:
            out[n] = {}
            for k in KS:
                v = np.array(curves[n][k])
                if len(v) < 5:
                    continue
                lo, hi = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
                out[n][str(k)] = {"ba": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(len(v)),
                                  "p_vs_half": float(stats.ttest_1samp(v, 0.5).pvalue)}
        b1 = out["epoch_z"]["1"]["ba"]
        out["prediction_from_k1"] = {str(k): float(stats.norm.cdf(np.sqrt(k) * stats.norm.ppf(b1)))
                                     for k in KS}
        res[par] = out
        print(par, {k: round(v["ba"], 3) for k, v in out["epoch_z"].items()},
              "pred", {k: round(v, 3) for k, v in out["prediction_from_k1"].items()},
              "pre_sd", {k: round(v["ba"], 3) for k, v in out["pre_sd"].items()},
              f"{time.time()-t0:.0f}s", flush=True)
    cm.write_json(OUT, {"provenance": {"experiment": "X03-2026-11", "ks": KS, "R": R, "norms": NORMS,
                                       "prediction": "BA(k) = Phi(sqrt(k) Phi^-1(BA(1)))"},
                        "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
