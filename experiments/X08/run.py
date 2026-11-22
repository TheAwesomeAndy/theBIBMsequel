"""X08-2026-11: centering controls for the SPL result (Reviewer 2, points 1-3).

Question: how much of the centering gain needs each test subject's complete
three-condition panel?

Exact relation, stated before the run: with three observations per subject,
leave-one-out centering x - mean(other two) equals 3/2 times full-panel
centering x - mean(all three). Excluding the target sample therefore leaves a
linear decoder's information unchanged; the control is reported to show it.

Test-side centering rules (training subjects are always full-panel centered,
the deployment case where the training cohort is complete):
  uncentered        : no centering (training stats only; equals deployment on one sample)
  full_panel        : SPL rule, subtract the mean of all 3 test observations
  loo               : subtract the mean of the subject's other 2 observations
  panel2            : each test subject contributes 2 of its 3 conditions,
                      chosen at random (5 draws), centered on those 2
  panel1            : each test subject contributes 1 condition; centering a
                      single observation gives the zero vector, so this row
                      uses the uncentered sample, i.e. no subject information

Scalings: epoch_z (SPL) and none. Encoder: raw EEG flattened (the SPL's
strongest linear row), the experiment_zero loader, 10-fold StratifiedGroupKFold
(seed 42), StandardScaler, LogisticRegression(C=0.1).
Null: within-subject label permutation for full_panel under none (50 perms).

Prediction, fixed before this run: loo equals full_panel; panel2 keeps most of
the gain; panel1 equals uncentered. The gain is then a within-subject contrast
effect that requires a multi-condition panel from the test subject.

Usage: SHAPE_DATA_DIR=... python experiments/X08/run.py
Output: outputs/X08/x08_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "X07"))
from sequel import common as cm  # noqa: E402
import run as x07  # noqa: E402

OUT = ROOT / "outputs" / "X08" / "x08_results.json"


def center_rows(F, g, rows, rule):
    out = F[rows].copy(); gg = g[rows]
    for s in np.unique(gg):
        m = gg == s
        if rule == "full_panel" or (rule == "panel" and m.sum() > 1):
            out[m] -= out[m].mean(0)
        elif rule == "loo":
            S = out[m].sum(0); n = m.sum()
            out[m] = out[m] - (S - out[m]) / (n - 1)
    return out


def run(F, y, g, rule, rng=None, yperm=None):
    yy = y if yperm is None else yperm
    pred_y, true_y = [], []
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g):
        Ftr = center_rows(F, g, tr, "full_panel") if rule != "uncentered" else F[tr]
        if rule in ("panel2", "panel1"):
            keep = []
            for s in np.unique(g[te]):
                idx = te[g[te] == s]
                keep += list(rng.choice(idx, 2 if rule == "panel2" else 1, replace=False))
            te = np.array(sorted(keep))
            Fte = center_rows(F, g, te, "panel") if rule == "panel2" else F[te]
        elif rule == "uncentered":
            Fte = F[te]
        else:
            Fte = center_rows(F, g, te, rule)
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(C=0.1, max_iter=3000, random_state=42).fit(sc.transform(Ftr), yy[tr])
        pred_y += list(clf.predict(sc.transform(Fte))); true_y += list(yy[te])
    return float(balanced_accuracy_score(true_y, pred_y))


def main():
    t0 = time.time()
    X0, y, g = x07.load_decimated()
    res = {}
    for mode in ["epoch_z", "none"]:
        F = x07.scale(X0, mode).reshape(len(X0), -1)
        r = {rule: run(F, y, g, rule) for rule in ["uncentered", "full_panel", "loo"]}
        for rule in ["panel2", "panel1"]:
            vals = [run(F, y, g, rule, rng=np.random.default_rng(d)) for d in range(5)]
            r[rule] = float(np.mean(vals)); r[rule + "_draws"] = vals
        res[mode] = r
        print(mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
    F = x07.scale(X0, "none").reshape(len(X0), -1)
    rng = np.random.default_rng(2026); null = []
    for _ in range(50):
        yp = y.copy()
        for s in np.unique(g):
            i = np.flatnonzero(g == s); yp[i] = rng.permutation(y[i])
        null.append(run(F, y, g, "full_panel", yperm=yp))
    res["null_full_panel_none"] = {"mean": float(np.mean(null)), "q975": float(np.percentile(null, 97.5)),
                                   "max": float(np.max(null))}
    print(res["null_full_panel_none"], flush=True)
    cm.write_json(OUT, {"provenance": cm.provenance("X08-2026-11", {"encoder": "raw EEG flattened",
                        "source": "experiment_zero.py protocol via experiments/X07"}),
                        "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
