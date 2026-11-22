"""X09-2026-11: which part of an epoch-own scaling carries the leak, and do the
exact transforms found in the X06 codebase audit leak?

Motivation. Proposition 1 has two routes by which post-stimulus samples reach a
pre-stimulus sample: the subtracted location statistic (epoch mean, minimum) and
the dividing scale statistic (epoch SD, range, amplitude percentile). X01 to X05
tested them only jointly (epoch_z). This run separates them and measures the
transforms that X06 found in public codebases.

Scalings (each per epoch and channel)
  none, pre_sd      leak-free anchors
  epoch_z           (x - m) / sd                       anchor
  epoch_center      x - m                              location route only
  epoch_sd          x / sd                             scale route only
  epoch_minmax      2 (x - min) / (max - min) - 1      DN3 MappingDeep1010 form
  epoch_p95         x / q95(|x|)                       BIOT form

Features on the pre-stimulus window
  pre_mean          channel means (first-order feature)
  pre_logsd         log SD of each channel over the window (second-order feature)
  reservoir_pre     onset-truncated ARSPI-Net reservoir (SHAPE only)

Predictions, fixed before this run
  P1  pre_mean leaks under every scaling with a location statistic taken from the
      whole epoch (epoch_z, epoch_center, epoch_minmax) and is at chance under
      none, pre_sd, epoch_sd, epoch_p95.
  P2  pre_logsd leaks under every scaling with a scale statistic taken from the
      whole epoch (epoch_z, epoch_sd, epoch_minmax, epoch_p95) and is at chance
      under none, pre_sd, epoch_center.
  P3  The onset-truncated reservoir, a nonlinear encoder, leaks under both routes.
  P4  The general identity (no baseline-correction assumption),
      mean_P z = (T1/T) (mean_P x - mean_Q x) / sd, holds to machine precision.

Run record (ERP CORE). On ERP CORE the pre-stimulus window of the features is
the whole interval that `pre_sd` divides by, so pre_logsd under pre_sd equals
log(1) = 0 for every trial and channel, up to rounding of order 1e-16. A
standardized readout then decodes rounding residue (the trap recorded for X02 in
outputs/X02/SUPERSEDED_RUNS.md). The first run computed that cell anyway. Its
values are kept in the output under `degenerate_value` and excluded from every
summary; the script now skips any feature that is constant to within 1e-9. On
SHAPE the feature window (samples 9 to 50) is a subset of the scaling window
(0 to 50), so the cell is well defined there.

Usage: SHAPE_DATA_DIR=... ERPCORE_DATA_DIR=... python experiments/X09/run.py [shape|erpcore]
Output: outputs/X09/x09_shape.json, outputs/X09/x09_erpcore.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

NORMS = ["none", "pre_sd", "epoch_z", "epoch_center", "epoch_sd", "epoch_minmax", "epoch_p95"]
PRE_LO = 9
OUTDIR = ROOT / "outputs" / "X09"


def scale(X, mode, pre):
    """X is (n, T, C); `pre` is a boolean mask or slice of pre-stimulus samples."""
    if mode == "none":
        return X.copy()
    if mode == "pre_sd":
        return X / X[:, pre, :].std(1, keepdims=True)
    m = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True)
    if mode == "epoch_z":
        return (X - m) / sd
    if mode == "epoch_center":
        return X - m
    if mode == "epoch_sd":
        return X / sd
    if mode == "epoch_minmax":
        lo = X.min(1, keepdims=True); hi = X.max(1, keepdims=True)
        return 2 * (X - lo) / (hi - lo) - 1
    if mode == "epoch_p95":
        return X / np.quantile(np.abs(X), 0.95, axis=1, keepdims=True)
    raise ValueError(mode)


def run_shape():
    t0 = time.time()
    raw, y, g = cm.load_shape_raw()
    Xb = cm.downsample(raw)
    pre = slice(0, cm.ONSET_DS); win = slice(PRE_LO, cm.ONSET_DS)
    T = Xb.shape[1]; T1 = T - cm.ONSET_DS
    # P4: general identity on the full pre-stimulus set
    Z = scale(Xb, "epoch_z", pre)
    lhs = Z[:, pre, :].mean(1)
    rhs = (T1 / T) * (Xb[:, pre, :].mean(1) - Xb[:, cm.ONSET_DS:, :].mean(1)) / Xb.std(1)
    ident = {"max_abs_error": float(np.abs(lhs - rhs).max()),
             "r": float(np.corrcoef(lhs.ravel(), rhs.ravel())[0, 1])}
    print("identity", ident, flush=True)
    preds, rates = {}, {}
    for n in NORMS:
        X = scale(Xb, n, pre)
        seg = X[:, win, :]
        preds[f"pre_mean|{n}"] = cm.oof_proba(seg.mean(1), y, g).argmax(1)
        preds[f"pre_logsd|{n}"] = cm.oof_proba(np.log(seg.std(1)), y, g).argmax(1)
        Fb, rate = cm.reservoir_bsc(X, t_start=PRE_LO, t_end=cm.ONSET_DS)
        rates[n] = rate
        preds[f"reservoir_pre|{n}"] = cm.oof_proba(Fb, y, g, block_pca_k=64).argmax(1)
        print(n, {k.split("|")[0]: round(cm.balanced_accuracy(y, v), 3) for k, v in preds.items() if k.endswith("|" + n)},
              f"rate {rate:.3f} {time.time() - t0:.0f}s", flush=True)
    summary, B, names = cm.subject_bootstrap(y, g, preds)
    contrasts = {f"{f}: {n} - none": cm.paired(B, names, f"{f}|{n}", f"{f}|none")
                 for f in ("pre_mean", "pre_logsd", "reservoir_pre") for n in NORMS if n != "none"}
    null = {}
    for f, n in (("pre_logsd", "epoch_sd"), ("pre_logsd", "none"), ("pre_mean", "epoch_center"), ("pre_mean", "epoch_sd")):
        seg = scale(Xb, n, pre)[:, win, :]
        F = seg.mean(1) if f == "pre_mean" else np.log(seg.std(1))
        d = cm.label_permutation_null(F, y, g, n_perm=200)
        obs = cm.balanced_accuracy(y, preds[f"{f}|{n}"])
        null[f"{f}|{n}"] = {"q95": float(np.percentile(d, 95)), "max": float(d.max()), "mean": float(d.mean()),
                            "p": float((1 + np.sum(d >= obs)) / 201)}
    cm.write_json(OUTDIR / "x09_shape.json", {
        "provenance": cm.provenance("X09-2026-11 (SHAPE)", {"norms": NORMS, "pre_window_ds": [PRE_LO, cm.ONSET_DS],
                                    "protocol": "as X01", "null": "200 within-subject label permutations, seed 42 partition"}),
        "general_identity": ident, "balanced_accuracy": summary, "contrasts_vs_none": contrasts,
        "firing_rate_reservoir_pre": rates, "permutation_null": null, "runtime_s": round(time.time() - t0, 1)})
    print(null)


def run_erpcore():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sequel import erpcore as ec
    N_PERM = 500

    def ba_flat(F, y, g, repeats, seed):
        acc = 0.0
        for r in range(repeats):
            pred = np.zeros(len(y), int)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(F, y, g):
                pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(F[tr], y[tr]).predict(F[te])
            acc += np.mean([(pred[y == c] == c).mean() for c in (0, 1)]) / repeats
        return float(acc)

    t0 = time.time(); res = {}
    for pi, par in enumerate(ec.CONTRASTS):
        rng = np.random.default_rng([909, pi])
        feats = {f"{f}|{n}": [] for f in ("pre_bins4", "pre_logsd") for n in NORMS}
        g, y2, err = [], [], 0.0
        for pid, f in ec.participants(par):
            x, times, y, _ = ec.load(f, par)
            if min((y == 0).sum(), (y == 1).sum()) == 0:
                continue
            pre = times < 0
            T = len(times); T1 = int((~pre).sum())
            z = scale(x, "epoch_z", pre)
            err = max(err, float(np.abs(z[:, pre].mean(1) - (T1 / T) * (x[:, pre].mean(1) - x[:, ~pre].mean(1)) / x.std(1)).max()))
            for n in NORMS:
                X = scale(x, n, pre)
                b4 = ec.pre_bins(X, times)                      # trials scaled, then averaged (pre-registered order)
                ls = np.log(X[:, pre].std(1))
                for c in (0, 1):
                    feats[f"pre_bins4|{n}"].append(b4[y == c].mean(0))
                    feats[f"pre_logsd|{n}"].append(ls[y == c].mean(0))
            g += [pid, pid]; y2 += [0, 1]
        g = np.array(g); y2 = np.array(y2); out = {"n_participants": len(g) // 2, "general_identity_max_abs_error": err}
        for k, v in feats.items():
            F = np.stack(v)
            if np.abs(F - F.mean(0)).max() < 1e-9:
                out[k] = {"degenerate": True, "reason": "feature constant by construction"}; continue
            obs = ba_flat(F, y2, g, 5, 0); null = []
            for _ in range(N_PERM):
                flip = rng.random(len(g) // 2) < 0.5
                yp = y2.reshape(-1, 2).copy(); yp[flip] = yp[flip][:, ::-1]
                null.append(ba_flat(F, yp.ravel(), g, 1, int(rng.integers(1e6))))
            null = np.array(null)
            out[k] = {"ba": obs, "p": float((1 + np.sum(null >= obs)) / (1 + N_PERM)), "null_q95": float(np.percentile(null, 95))}
            print(f"{par:5s} {k:24s} {obs:.3f} p={out[k]['p']:.3f} {time.time() - t0:.0f}s", flush=True)
        res[par] = out
        cm.write_json(OUTDIR / "x09_erpcore.json", {
            "provenance": {"experiment": "X09-2026-11 (ERP CORE)", "norms": NORMS, "n_perm": N_PERM,
                           "features": "pre_bins4: four 12-sample pre-stimulus bin means x 28 channels (112); pre_logsd: log SD over the pre-stimulus interval x 28 channels; trials scaled then averaged per participant and condition",
                           "cv": "participant-grouped 5-fold x5, standardized L2 logistic; null: within-participant label swaps, one repeat each"},
            "results": res, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    which = sys.argv[1:] or ["shape", "erpcore"]
    if "shape" in which:
        run_shape()
    if "erpcore" in which:
        run_erpcore()
