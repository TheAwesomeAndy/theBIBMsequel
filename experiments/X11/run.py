"""X11-2026-11 (Paper B): panel centering against alignment alternatives, an
assignment readout, and a second dataset.

Motivation. X08 established that the subject-centering gain requires an
unlabeled multi-condition panel from the test participant. Under the additive
model x_{s,c} = a_s + mu_c + e_{s,c}, subtracting the panel mean removes a_s
exactly when the panel is complete and balanced. Two questions follow.
(1) Other transductive alignments use the same unlabeled panel (Euclidean
alignment, He and Wu 2020; Riemannian recentering, Zanini et al. 2018). They
act on second-order statistics. Does a second-order alignment remove a
first-order subject offset? The model predicts it does not.
(2) A complete balanced panel also fixes the label multiset of each test
participant. The readout that uses this design knowledge assigns the panel's
labels jointly (one observation per condition). The model predicts a further
gain from joint assignment over independent argmax.

Part A, SHAPE (unscaled input, the X07/X08 loader and protocol: 10-fold
StratifiedGroupKFold seed 42, StandardScaler, LogisticRegression C=0.1).
Every transform is applied to training and test participants alike and uses
only that participant's own unlabeled observations.
  uncentered      flattened epoch
  panel_center    subtract the participant's panel mean
  subject_z       panel_center, then divide each feature by the participant's
                  panel SD (three observations)
  ea              Euclidean alignment: X R_s^{-1/2}, R_s the participant's mean
                  spatial covariance
  ea_center       ea followed by panel_center
  tangent         log-Euclidean tangent vectors of each epoch's spatial
                  covariance at the training-cohort mean
  tangent_recenter  the same after whitening each covariance by the
                  participant's own mean covariance (recentering)
  panel_center_assign  panel_center features, labels of each test participant
                  assigned jointly by maximum total log-probability
Part B, ERP CORE (five paradigms, participant-condition averages, unscaled,
4-sample bins x 28 channels): uncentered and panel_center, participant-grouped
5-fold x5; within-participant label-swap null (500) for panel_center.

Predictions, fixed before this run: ea and tangent_recenter stay near
uncentered and tangent; ea_center matches panel_center; panel_center_assign
is at or above panel_center; panel_center exceeds uncentered in every
ERP CORE paradigm.

Run record. The first SHAPE run used scipy fractional_matrix_power and logm on
the spatial covariances, which are rank deficient by one (common reference),
and logm reported an unbounded error for the two tangent rows. Those two rows
were discarded. This version uses a pseudo-inverse square root and a ridge of
1e-3 times the mean eigenvalue before the logarithm. A second attempt stopped
with non-finite tangent features, because the pseudo-inverse reference removed
one direction and left a zero eigenvalue under the logarithm; the tangent rows
now use the full inverse of the regularized reference. The cause of the
non-finite values was one participant with an all-zero epoch (the X07 loader
applies no exclusion). That participant is now excluded from every row, so this
run has one participant fewer than X07 and X08. The first run's other rows
(uncentered 0.678, panel_center 0.912, subject_z 0.904, ea 0.822, ea_center
0.904, panel_center_assign 0.939) do not involve the logarithm. The prediction
that ea stays near uncentered was not supported in either run.

Usage: SHAPE_DATA_DIR=... ERPCORE_DATA_DIR=... python experiments/X11/run.py [shape|erpcore]
Output: outputs/X11/x11_shape.json, outputs/X11/x11_erpcore.json
"""
from __future__ import annotations

import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "X07"))
from sequel import common as cm  # noqa: E402

OUTDIR = ROOT / "outputs" / "X11"


def by_subject(g):
    return [np.flatnonzero(g == s) for s in np.unique(g)]


def panel_center(F, g):
    out = F.copy()
    for i in by_subject(g):
        out[i] -= out[i].mean(0)
    return out


def subject_z(F, g):
    out = panel_center(F, g)
    for i in by_subject(g):
        sd = out[i].std(0); out[i] /= np.where(sd > 0, sd, 1.0)
    return out


def covs(X):
    return np.einsum("ntc,ntd->ncd", X, X) / X.shape[1]


def inv_sqrt(C, tol=1e-8):
    """Pseudo-inverse square root. The delivered epochs are rank deficient by one
    (common reference), so the null direction is dropped instead of inverted.
    tol=0 gives the full inverse square root for ridge-regularized input."""
    w, V = np.linalg.eigh(C); k = w > tol * w.max()
    return (V[:, k] / np.sqrt(w[k])) @ V[:, k].T


def shrink(C, lam=1e-3):
    """Ridge toward the identity so the matrix logarithm exists (rank-deficient input)."""
    return C + lam * np.trace(C, axis1=-2, axis2=-1)[..., None, None] / C.shape[-1] * np.eye(C.shape[-1])


def ea(X, g):
    out = X.copy(); C = covs(X)
    for i in by_subject(g):
        out[i] = X[i] @ inv_sqrt(C[i].mean(0))
    return out


def tangent(C, ref):
    W = inv_sqrt(ref, tol=0.0); iu = np.triu_indices(C.shape[1])
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)); out = []
    for c in C:
        w, V = np.linalg.eigh(W @ c @ W)
        out.append(((V * np.log(w)) @ V.T)[iu] * coef)
    return np.stack(out)


def recenter(C, g):
    out = C.copy()
    for i in by_subject(g):
        W = inv_sqrt(C[i].mean(0), tol=0.0)      # C is ridge-regularized, so the full inverse exists
        out[i] = W @ C[i] @ W
    return out


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def oof(F, y, g, cov_mode=None, C=None, assign=False):
    """Out-of-fold predictions. cov_mode: tangent features need a training-fold reference."""
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g):
        if cov_mode:
            ref = C[tr].mean(0); Ftr, Fte = tangent(C[tr], ref), tangent(C[te], ref)
        else:
            Ftr, Fte = F[tr], F[te]
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(C=0.1, max_iter=3000, random_state=42).fit(sc.transform(Ftr), y[tr])
        if not assign:
            pred[te] = clf.predict(sc.transform(Fte)); continue
        L = clf.predict_log_proba(sc.transform(Fte)); K = L.shape[1]
        for i in by_subject(g[te]):
            best = max(permutations(range(K), len(i)), key=lambda p: sum(L[i[j], p[j]] for j in range(len(i))))
            pred[te[i]] = clf.classes_[list(best)]
    return pred


def boot(y, g, preds, n=4000, seed=0):
    out, B, names = cm.subject_bootstrap(y, g, preds, n_boot=n, seed=seed)
    return out, B, names


def run_shape():
    import run as x07
    t0 = time.time()
    X, y, g = x07.load_decimated()
    X = x07.scale(X, "none")
    # A participant with an all-zero epoch has no spatial covariance. Excluded from every row.
    bad = np.unique(g[np.trace(covs(X), axis1=1, axis2=2) <= 0])
    keep = ~np.isin(g, bad); X, y, g = X[keep], y[keep], g[keep]
    print(f"excluded {len(bad)} participant(s) with a zero-variance epoch; {len(np.unique(g))} remain", flush=True)
    flat = lambda A: A.reshape(len(A), -1)
    Xea = ea(X, g); C = shrink(covs(X))
    preds = {
        "uncentered": oof(flat(X), y, g),
        "panel_center": oof(panel_center(flat(X), g), y, g),
        "subject_z": oof(subject_z(flat(X), g), y, g),
        "ea": oof(flat(Xea), y, g),
        "ea_center": oof(panel_center(flat(Xea), g), y, g),
        "tangent": oof(None, y, g, cov_mode=True, C=C),
        "tangent_recenter": oof(None, y, g, cov_mode=True, C=shrink(recenter(C, g))),
        "panel_center_assign": oof(panel_center(flat(X), g), y, g, assign=True),
    }
    print({k: round(bal(y, v), 3) for k, v in preds.items()}, f"{time.time() - t0:.0f}s", flush=True)
    summary, B, names = boot(y, g, preds)
    contrasts = {f"{a} - {b}": cm.paired(B, names, a, b) for a, b in [
        ("panel_center", "uncentered"), ("ea", "uncentered"), ("ea_center", "panel_center"),
        ("subject_z", "panel_center"), ("tangent_recenter", "tangent"),
        ("panel_center_assign", "panel_center"), ("panel_center", "ea")]}
    # additive-model variance fractions of the flattened unscaled epoch (per feature, then summed)
    F = flat(X); subs = by_subject(g)
    gm = F.mean(0); a = np.stack([F[i].mean(0) for i in subs]); m = np.stack([F[y == c].mean(0) for c in np.unique(y)])
    ss_tot = ((F - gm) ** 2).sum(); ss_s = 3 * ((a - gm) ** 2).sum(); ss_c = len(subs) * ((m - gm) ** 2).sum()
    var = {"subject": float(ss_s / ss_tot), "condition": float(ss_c / ss_tot), "residual": float(1 - (ss_s + ss_c) / ss_tot)}
    cm.write_json(OUTDIR / "x11_shape.json", {
        "provenance": cm.provenance("X11-2026-11 (SHAPE)", {"loader": "experiments/X07 (212 participants, decimate by 4, unscaled)",
                                    "protocol": "10-fold StratifiedGroupKFold seed 42, StandardScaler, LogisticRegression C=0.1; 4000 participant bootstraps of out-of-fold predictions"}),
        "n_obs": int(len(y)), "n_subjects": len(subs), "n_excluded_zero_variance": int(len(bad)), "balanced_accuracy": summary, "contrasts": contrasts,
        "variance_fractions_flattened_epoch": var, "runtime_s": round(time.time() - t0, 1)})
    for k, v in summary.items():
        print(f"{k:22s} {v['ba']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    print(contrasts, var)


def run_erpcore():
    from sequel import erpcore as ec
    N_PERM = 500

    def cv(F, y, g, repeats, seed):
        P = np.zeros((repeats, len(y)), int)
        for r in range(repeats):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(F, y, g):
                sc = StandardScaler().fit(F[tr])
                P[r, te] = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(F[tr]), y[tr]).predict(sc.transform(F[te]))
        return P

    t0 = time.time(); res = {}
    for pi, par in enumerate(ec.CONTRASTS):
        rng = np.random.default_rng([1111, pi]); F, g, y = [], [], []
        for pid, f in ec.participants(par):
            x, times, lab, _ = ec.load(f, par)
            if min((lab == 0).sum(), (lab == 1).sum()) == 0:
                continue
            for c in (0, 1):
                a = x[lab == c].mean(0); nb = a.shape[0] // 4
                F.append(a[:nb * 4].reshape(nb, 4, -1).mean(1).ravel()); g.append(pid); y.append(c)
        F = np.stack(F); g = np.array(g); y = np.array(y); Fc = panel_center(F, g)
        out = {"n_participants": len(g) // 2}
        subs = np.unique(g); brng = np.random.default_rng(0)
        for name, feat in (("uncentered", F), ("panel_center", Fc)):
            P = cv(feat, y, g, 5, 0); acc = (P == y).mean(0)       # per-observation accuracy over repeats
            per_sub = np.array([acc[g == s].mean() for s in subs])
            bs = [per_sub[brng.integers(0, len(subs), len(subs))].mean() for _ in range(4000)]
            out[name] = {"ba": float(per_sub.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        null = []
        for _ in range(N_PERM):
            flip = rng.random(len(g) // 2) < 0.5
            yp = y.reshape(-1, 2).copy(); yp[flip] = yp[flip][:, ::-1]; yp = yp.ravel()
            null.append(float((cv(Fc, yp, g, 1, int(rng.integers(1e6)))[0] == yp).mean()))
        null = np.array(null)
        out["panel_center"]["p"] = float((1 + np.sum(null >= out["panel_center"]["ba"])) / (1 + N_PERM))
        out["panel_center"]["null_q95"] = float(np.percentile(null, 95))
        res[par] = out
        print(par, out, f"{time.time() - t0:.0f}s", flush=True)
        cm.write_json(OUTDIR / "x11_erpcore.json", {
            "provenance": {"experiment": "X11-2026-11 (ERP CORE)", "features": "participant-condition averages, unscaled, 4-sample bin means x 28 channels over the full epoch",
                           "cv": "participant-grouped 5-fold x5, StandardScaler, LogisticRegression C=0.1; balanced design so accuracy equals balanced accuracy; 4000 participant bootstraps",
                           "null": f"{N_PERM} within-participant label swaps, one repeat each, panel_center"},
            "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    which = sys.argv[1:] or ["shape", "erpcore"]
    if "shape" in which:
        run_shape()
    if "erpcore" in which:
        run_erpcore()
