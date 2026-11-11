#!/usr/bin/env python3
"""Camera-ready experiments E1 and E2 (BIBM 2026 Doctoral Forum, reviewer response).

E1 - spectral-radius sweep. The recurrent matrix W is rescaled to rho in RHO_GRID
     (rho = 0 removes recurrence: a feedforward LIF layer). For each rho the BSC6
     code is regenerated and evaluated with train-only PCA (fit per fold on the
     training subjects) under the source protocol, giving subject-level BA [95% CI]
     for: clean; 30% feature-coordinate dropout (zero and train-mean fill); and 30%
     signal-level removal (dropped channel zeroed before the PCA mixing). Paired
     subject-bootstrap differences against rho = 0.9 are reported.
     A label-free dynamical order parameter is measured on a fine rho grid by
     damage spreading: two copies of the reservoir receive the same ERP drive; at
     t = T_START one neuron's spike state is flipped in one copy; the normalized
     Hamming distance between the copies' spike vectors is averaged over the
     analysis window. Ordered dynamics forget the flip (distance -> 0); chaotic
     dynamics amplify it.

E2 - feature-coordinate experiment (four fills, 10-50% dropout) with the
     reservoir embedding fit by train-only PCA in every fold (instead of the pooled
     basis). Band power is computed with the documented Welch estimator on the
     analysed signal (as in the signal-level experiment); ERP-window is unchanged.
     Paired subject-level contrasts of each fill against zero-fill are reported at
     every dropout level (`fill_minus_zero`, used by Fig. 3).

Usage: python analysis/experiment_camera_ready.py [all|e1|e2]

Protocol (unchanged from the source): StratifiedGroupKFold(5, shuffle) x seeds
42-46, subject groups, train-only StandardScaler, balanced L2 logistic readout;
out-of-fold probabilities averaged per observation across the five partitions;
subject-level bootstrap (211 subjects, three observations each), n_boot = 4000.

Reads the restricted SHAPE pickle locally; writes only aggregate JSON to
outputs/aggregate/. Never writes subject-level data.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis_subject_bootstrap as RB  # noqa: E402  (fills, bootstrap, blocks)
import experiment2_rawsignal as E2S  # noqa: E402  (band-power estimator)

N_RES, BETA, THETA, SEED = 256, 0.05, 0.5, 42
BSC_N_BINS, T_START, T_END, PCA_K = 6, 10, 70, 64
SEEDS = [42, 43, 44, 45, 46]
RHO_GRID = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
RHO_FINE = [round(r, 2) for r in np.arange(0.0, 2.01, 0.1)]
FOCUS = 0.3
OUT = Path(__file__).resolve().parents[1] / "outputs" / "aggregate"


def weights(rho):
    """Source initialization (Xavier-uniform, seed 42); W rescaled to spectral radius rho."""
    rng = np.random.RandomState(SEED)
    lin = np.sqrt(6.0 / (1 + N_RES)); W_in = rng.uniform(-lin, lin, (N_RES, 1))
    lrec = np.sqrt(6.0 / (2 * N_RES)); W = rng.uniform(-lrec, lrec, (N_RES, N_RES))
    W *= rho / np.abs(np.linalg.eigvals(W)).max()
    return W_in, W


def bsc6(X_ds, rho):
    """BSC6 for every (obs, channel) drive: returns (N, 34, 1536) and mean firing rate."""
    N, T, n_ch = X_ds.shape; C = N * n_ch
    S = np.transpose(X_ds, (1, 0, 2)).reshape(T, C)
    W_in, W = weights(rho); win0 = W_in[:, 0][:, None]
    mem = np.zeros((N_RES, C)); spk = np.zeros((N_RES, C))
    counts = np.zeros((N_RES, BSC_N_BINS, C), dtype=np.float32)
    binw = (T_END - T_START) // BSC_N_BINS
    for t in range(T):
        mem = (1.0 - BETA) * mem * (1.0 - spk) + win0 * S[t][None, :] + W @ spk
        spk = (mem >= THETA).astype(np.float64)
        mem = np.maximum(mem - spk * THETA, 0.0)
        if T_START <= t < T_END:
            counts[:, (t - T_START) // binw, :] += spk
    rate = float(counts.sum() / (N_RES * C * (T_END - T_START)))
    return counts.reshape(N_RES * BSC_N_BINS, C).T.reshape(N, n_ch, -1), rate


def damage(X_ds, rho, n_drive=400, rng_seed=0):
    """Damage-spreading order parameter: mean normalized Hamming distance after a one-spike flip."""
    N, T, n_ch = X_ds.shape
    rng = np.random.default_rng(rng_seed)
    cols = rng.choice(N * n_ch, n_drive, replace=False)
    S = np.transpose(X_ds, (1, 0, 2)).reshape(T, N * n_ch)[:, cols]
    flip = rng.integers(0, N_RES, n_drive)
    W_in, W = weights(rho); win0 = W_in[:, 0][:, None]
    mem = [np.zeros((N_RES, n_drive)) for _ in range(2)]
    spk = [np.zeros((N_RES, n_drive)) for _ in range(2)]
    dist = []
    for t in range(T_END):
        for k in range(2):
            mem[k] = (1.0 - BETA) * mem[k] * (1.0 - spk[k]) + win0 * S[t][None, :] + W @ spk[k]
            spk[k] = (mem[k] >= THETA).astype(np.float64)
            mem[k] = np.maximum(mem[k] - spk[k] * THETA, 0.0)
        if t == T_START:
            spk[1][flip, np.arange(n_drive)] = 1.0 - spk[1][flip, np.arange(n_drive)]
        if t > T_START + 10:
            dist.append(np.abs(spk[0] - spk[1]).mean(axis=0))
    return float(np.mean(dist))


def fit_pred(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="lbfgs", random_state=seed).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def fold_embeddings(B, y, g):
    """Train-only PCA per (seed, fold): yields seed, fold, tr, te, E (N,34,64), v0 (64,)."""
    N, n_ch, D = B.shape
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((N, 1)), y, groups=g)):
            pca = PCA(n_components=PCA_K, random_state=SEED).fit(B[tr].reshape(-1, D))
            E = pca.transform(B.reshape(-1, D)).reshape(N, n_ch, PCA_K)
            v0 = pca.transform(np.zeros((1, D)))[0]      # embedding of a removed (silent) channel
            yield seed, fold, tr, te, E, v0


def e1_conditions(B, y, g, cache=None):
    """Clean, feature-coordinate 30% (zero, mean), signal-level 30% for one rho."""
    N, n_ch, _ = B.shape; classes = np.unique(y)
    acc = {k: np.zeros((N, classes.size)) for k in ("clean", "fc_zero", "fc_mean", "signal")}
    for seed, fold, tr, te, E, v0 in fold_embeddings(B, y, g):
        if cache is not None:
            cache.append((seed, fold, tr, te, E.astype(np.float32)))
        Xtr = E[tr].reshape(len(tr), -1)
        rng = np.random.default_rng(1000 * seed + fold)         # same drop draw as the source audit
        drop = rng.choice(n_ch, int(round(FOCUS * n_ch)), replace=False)
        acc["clean"][te] += fit_pred(Xtr, y[tr], E[te].reshape(len(te), -1), seed)
        for fill in ("zero", "mean"):
            Ete = E[te].copy()
            Ete[:, drop, :] = 0.0 if fill == "zero" else E[tr][:, drop, :].mean(axis=0)
            acc[f"fc_{fill}"][te] += fit_pred(Xtr, y[tr], Ete.reshape(len(te), -1), seed)
        Ete = E[te].copy(); Ete[:, drop, :] = v0
        acc["signal"][te] += fit_pred(Xtr, y[tr], Ete.reshape(len(te), -1), seed)
    return {k: classes[np.argmax(v, axis=1)] for k, v in acc.items()}


def run_e1(X_ds, y, g, ref_rho=0.9):
    res = {"rho_grid": RHO_GRID, "BA": {}, "paired_vs_rho0.9": {}, "firing_rate": {},
           "damage": {}}
    preds, cache09 = {}, []
    for rho in RHO_GRID:
        t0 = time.time()
        B, rate = bsc6(X_ds, rho)
        preds[rho] = e1_conditions(B, y, g, cache=cache09 if rho == ref_rho else None)
        pt, ci, _ = RB.subject_bootstrap(y, g, preds[rho])
        res["BA"][str(rho)] = {k: {"BA": pt[k], "ci95": list(ci[k])} for k in pt}
        res["firing_rate"][str(rho)] = rate
        print(f"[E1 rho={rho}] rate={rate:.3f} " + " | ".join(
            f"{k}={pt[k]:.3f}[{ci[k][0]:.3f},{ci[k][1]:.3f}]" for k in pt)
            + f"  ({time.time() - t0:.0f}s)", flush=True)
    for rho in RHO_GRID:
        if rho == ref_rho:
            continue
        res["paired_vs_rho0.9"][str(rho)] = {}
        for k in ("clean", "fc_mean", "signal"):
            m, lo, hi, p = RB.paired_diff_ci(y, g, preds[rho][k], preds[ref_rho][k])
            res["paired_vs_rho0.9"][str(rho)][k] = {"mean_diff": m, "ci95": [lo, hi]}
    for rho in RHO_FINE:
        res["damage"][str(rho)] = damage(X_ds, rho)
    print("[E1 damage] " + " ".join(f"{r}:{res['damage'][str(r)]:.3f}" for r in RHO_FINE), flush=True)
    return res, cache09


def run_e2(d, y, g, cache09):
    """Feature-coordinate experiment with train-only PCA for the reservoir (rho = 0.9)."""
    N = len(y); classes = np.unique(y); n_ch = 34
    # Band power from the documented estimator on the analysed signal (Welch, nperseg 256,
    # bands 1-4/4-8/8-13/13-30/30-45 Hz, trapezoid), identical to the signal-level experiment.
    # The pickle's conv_feats are NOT used: they cannot be reproduced from X_ds with that
    # estimator (audit, camera-ready), so all band-power results now share one definition.
    Xds = d["X_ds"].astype(float)
    band = E2S.bandpower_all(Xds)
    erp = np.zeros((N, n_ch, 3))
    for w, (a, b) in enumerate(RB.ERP_WINDOWS_MS):
        erp[:, :, w] = Xds[:, int(round(a / 1000 * RB.FS)):int(round(b / 1000 * RB.FS)), :].mean(axis=1)
    fixed = {"Band-power": (band.reshape(N, -1), band.shape[1:]),
             "ERP-window": (erp.reshape(N, -1), erp.shape[1:])}
    preds = {}
    for frac in [0.0] + RB.DROP_LEVELS:
        for fill in RB.FILLS:
            if frac == 0.0 and fill != "zero":
                continue
            for name, (X, ch) in fixed.items():
                preds[(name, frac, fill)] = RB.run_oof(X, ch, y, g, frac, fill)
            acc = np.zeros((N, classes.size))
            for seed, fold, tr, te, E in cache09:
                X = E.reshape(N, -1).astype(np.float64)
                Xtr, Xte = X[tr], X[te]
                if frac > 0:
                    rng = np.random.default_rng(1000 * seed + fold)
                    drop, keep, _ = RB.drop_channels((n_ch, PCA_K), frac, rng)
                    Xte = RB.fill_test(Xte, Xtr, (n_ch, PCA_K), drop, keep, fill)
                acc[te] += fit_pred(Xtr, y[tr], Xte, seed)
            preds[("Reservoir", frac, fill)] = classes[np.argmax(acc, axis=1)]
        print(f"[E2 {int(frac * 100)}%] done", flush=True)
    names = ["Band-power", "ERP-window", "Reservoir"]
    out = {"clean_BA": {}, "dropout_30": {}, "paired_ERPwindow_minus_Reservoir_30": {},
           "mean_minus_zero_30": {}, "curve_BA": {}}
    pt, ci, _ = RB.subject_bootstrap(y, g, {n: preds[(n, 0.0, "zero")] for n in names})
    out["clean_BA"] = {n: {"point": pt[n], "ci95": list(ci[n])} for n in names}
    for fill in RB.FILLS:
        pm = {n: preds[(n, FOCUS, fill)] for n in names}
        p2, c2, _ = RB.subject_bootstrap(y, g, pm)
        out["dropout_30"][fill] = {n: {"drop_BA": p2[n], "drop_ci95": list(c2[n])} for n in names}
        m, lo, hi, pgt = RB.paired_diff_ci(y, g, pm["ERP-window"], pm["Reservoir"])
        out["paired_ERPwindow_minus_Reservoir_30"][fill] = {"mean_diff": m, "ci95": [lo, hi],
                                                             "P(ERPwindow>Reservoir)": pgt}
    for n in names:
        m, lo, hi, _ = RB.paired_diff_ci(y, g, preds[(n, FOCUS, "mean")], preds[(n, FOCUS, "zero")])
        out["mean_minus_zero_30"][n] = {"mean_diff": m, "ci95": [lo, hi]}
        out["curve_BA"][n] = {fill: {str(int(f * 100)): float(balanced_accuracy_score(
            y, preds[(n, f, fill)])) for f in RB.DROP_LEVELS} for fill in RB.FILLS}
    # paired subject-level contrast of every fill against zero-fill, at every dropout level
    out["fill_minus_zero"] = {n: {fill: {} for fill in RB.FILLS if fill != "zero"} for n in names}
    for n in names:
        for fill in out["fill_minus_zero"][n]:
            for f in RB.DROP_LEVELS:
                m, lo, hi, _ = RB.paired_diff_ci(y, g, preds[(n, f, fill)], preds[(n, f, "zero")])
                out["fill_minus_zero"][n][fill][str(int(f * 100))] = {"mean_diff": m, "ci95": [lo, hi]}
    return out


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X_ds = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    proto = ("StratifiedGroupKFold(5,shuffle) x seeds 42-46; train-only StandardScaler; balanced L2 "
             "logreg; train-only PCA-64 per fold; OOF probabilities averaged over partitions; "
             "subject-level bootstrap n_boot=%d" % RB.N_BOOT)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"          # all | e1 | e2
    cache09 = []
    if part in ("all", "e1"):
        e1, cache09 = run_e1(X_ds, y, g)
        e1["protocol"] = proto
        json.dump(e1, open(OUT / "e1_rho_sweep.json", "w"), indent=2)
        print("[out] wrote e1_rho_sweep.json", flush=True)
        if part == "e1":
            return 0
    if not cache09:                                               # E2 alone: rebuild rho = 0.9 folds
        B, _ = bsc6(X_ds, 0.9)
        cache09 = [(s_, f_, tr, te, E.astype(np.float32)) for s_, f_, tr, te, E, _v in fold_embeddings(B, y, g)]
    e2 = run_e2(d, y, g, cache09)
    e2["protocol"] = proto + "; reservoir rho=0.9"
    json.dump(e2, open(OUT / "e2_trainonly_pca_fills.json", "w"), indent=2)
    print("[out] wrote e2_trainonly_pca_fills.json", flush=True)
    for fill, r in e2["paired_ERPwindow_minus_Reservoir_30"].items():
        print(f"[E2 30% {fill}] ERP-Res={r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    for n, r in e2["mean_minus_zero_30"].items():
        print(f"[E2 mean-zero 30%] {n}: {r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
