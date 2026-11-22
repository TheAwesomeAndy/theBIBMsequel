"""X12-2026-11 (Paper B): the reference-size law for subject centering, on
single trials of ERP CORE.

Model. Observation i of participant s: x = a_s + mu_c + e, with a_s the
participant offset (covariance S_a) and e the trial noise (covariance S_e).
Centering on the mean of m unlabeled reference observations of the same
participant (not containing the test observation, class composition equal to
the design's) leaves noise covariance S_e (1 + 1/m) and no participant term.
Along a discriminant direction the squared separation is therefore

    D2(m) = D2(inf) * m / (m + 1),      D2(0) = uncentered separation.

Prediction, fixed before this run: balanced accuracy rises with m and
saturates quickly; in probit units, Phi^-1(BA(m))^2 / Phi^-1(BA(inf))^2
follows m / (m + 1) once m is large enough that centering beats the
uncentered readout; Euclidean alignment, a second-order transform, does not
remove the offset by itself.

Design. Cross-participant single-trial decoding, five stimulus-locked
ERP CORE paradigms. Features: unscaled post-stimulus epoch, 8-sample bin
means x 28 channels. Participant-grouped 5-fold cross-validation, one seed,
StandardScaler and L2 logistic regression (C=0.1, class-balanced) fitted on
training participants. Training participants are always centered on their
own full trial mean (label-free).
Test-side rules
  uncentered           training also uncentered
  center_all           subtract the participant's mean over all its trials
                       (transductive, label-free)
  center_ref_m         m in {1, 2, 4, 8, 16, 32}: the reference mean is
                       taken over m trials drawn at random (label-free, so the
                       class composition is the participant's own) and those
                       trials are removed from the test set
  center_causal        running mean over the participant's preceding trials
                       in recorded order (first trial left uncentered)
  ea, ea_center_all    Euclidean alignment with the participant's mean spatial
                       covariance over all its trials, without and with
                       center_all
Uncertainty: participant bootstrap (2000) of per-participant balanced accuracy.

Run record. A first run used m up to 64 and required 128 trials per participant.
It completed N170 and P3 and stopped at N400, where no participant has 128
retained trials. That partial output was discarded and the design changed to
m <= 32 with a 64-trial requirement, applied to all five paradigms.

Usage: ERPCORE_DATA_DIR=... python experiments/X12/run.py [PARADIGM ...]
Output: outputs/X12/x12_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402
from sequel import erpcore as ec  # noqa: E402

MS = [1, 2, 4, 8, 16, 32]
OUT = ROOT / "outputs" / "X12" / "x12_results.json"


def inv_sqrt(C, tol=1e-8):
    """Pseudo-inverse square root (the epochs are rank deficient by one)."""
    w, V = np.linalg.eigh(C); k = w > tol * w.max()
    return (V[:, k] / np.sqrt(w[k])) @ V[:, k].T


def feats(x, times):
    q = x[:, times >= 0]; nb = q.shape[1] // 8
    return q[:, :nb * 8].reshape(len(q), nb, 8, -1).mean(2).reshape(len(q), -1)


def pba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in (0, 1)]))


def main(only=None):
    t0 = time.time()
    res = json.loads(OUT.read_text())["results"] if OUT.exists() else {}
    for pi, par in enumerate(ec.CONTRASTS):
        if par in res or (only and par not in only):
            continue
        rng = np.random.default_rng([1212, pi])
        F, Fea, y, g = [], [], [], []
        for pid, f in ec.participants(par):
            x, times, lab, _ = ec.load(f, par)
            if min((lab == 0).sum(), (lab == 1).sum()) < 5 or len(lab) < 2 * max(MS):
                continue
            R = np.einsum("ntc,ntd->cd", x, x) / (x.shape[0] * x.shape[1])
            F.append(feats(x, times)); Fea.append(feats(x @ inv_sqrt(R), times))
            y.append(lab); g.append(np.full(len(lab), pid))
        subs = [gi[0] for gi in g]; n_s = len(subs)
        Fc = [f - f.mean(0) for f in F]; Feac = [f - f.mean(0) for f in Fea]
        rules = ["uncentered", "center_all", "center_causal", "ea", "ea_center_all"] + [f"center_ref_{m}" for m in MS]
        per = {r: np.full(n_s, np.nan) for r in rules}
        for tr, te in GroupKFold(5).split(np.zeros(n_s), groups=np.arange(n_s)):
            def fit(parts):
                X = np.concatenate([parts[i] for i in tr]); Y = np.concatenate([y[i] for i in tr])
                sc = StandardScaler().fit(X)
                return sc, LogisticRegression(C=0.1, max_iter=500, class_weight="balanced").fit(sc.transform(X), Y)
            models = {"raw": fit(F), "cen": fit(Fc), "ea": fit(Fea), "eacen": fit(Feac)}

            def score(key, X, Y):
                sc, clf = models[key]; return pba(Y, clf.predict(sc.transform(X)))
            for i in te:
                per["uncentered"][i] = score("raw", F[i], y[i])
                per["center_all"][i] = score("cen", Fc[i], y[i])
                per["ea"][i] = score("ea", Fea[i], y[i])
                per["ea_center_all"][i] = score("eacen", Feac[i], y[i])
                n = len(y[i]); run = np.cumsum(F[i], 0)[:-1] / np.arange(1, n)[:, None]
                per["center_causal"][i] = score("cen", np.vstack([F[i][:1], F[i][1:] - run]), y[i])
                for m in MS:
                    vals = []
                    for _ in range(5):
                        ref = rng.choice(n, m, replace=False); keep = np.setdiff1d(np.arange(n), ref)
                        vals.append(score("cen", F[i][keep] - F[i][ref].mean(0), y[i][keep]))
                    per[f"center_ref_{m}"][i] = np.mean(vals)
        brng = np.random.default_rng(0); out = {"n_participants": n_s, "n_trials": int(sum(map(len, y)))}
        idx = brng.integers(0, n_s, (2000, n_s))
        for r in rules:
            b = per[r][idx].mean(1)
            out[r] = {"ba": float(per[r].mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        d = per["center_all"] - per["uncentered"]; b = d[idx].mean(1)
        out["gain_center_all"] = {"diff": float(d.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        d = per["ea"] - per["uncentered"]; b = d[idx].mean(1)
        out["gain_ea"] = {"diff": float(d.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        zinf = norm.ppf(out["center_all"]["ba"])
        out["law"] = {str(m): {"observed_ba": out[f"center_ref_{m}"]["ba"],
                               "predicted_ba": float(norm.cdf(zinf * np.sqrt(m / (m + 1))))} for m in MS}
        res[par] = out
        print(par, {k: (round(v["ba"], 3) if isinstance(v, dict) and "ba" in v else v) for k, v in out.items() if k != "law"},
              f"{time.time() - t0:.0f}s", flush=True)
        print("   law", {m: (round(v["observed_ba"], 3), round(v["predicted_ba"], 3)) for m, v in out["law"].items()}, flush=True)
        cm.write_json(OUT, {"provenance": {"experiment": "X12-2026-11", "ms": MS,
                                           "features": "unscaled post-stimulus 8-sample bin means x 28 channels",
                                           "cv": "participant GroupKFold(5), StandardScaler, class-balanced L2 logistic C=0.1",
                                           "law": "predicted BA(m) = Phi(Phi^-1(BA(center_all)) * sqrt(m/(m+1)))"},
                            "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main(sys.argv[1:] or None)
