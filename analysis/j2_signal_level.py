#!/usr/bin/env python3
"""J2: signal-level perturbations for the fixed encoders on the external TCRZEM cohort.

Conference conditions (perturbations.py, identical draws for every encoder):
  clean; electrode removal at 10/30/50% (zero trace); additive white noise at 5 dB SNR;
  uniform +/-50 ms alignment jitter.
New conditions:
  spline_{10,30,50}: the same removed electrodes, repaired by spherical-spline
      interpolation from the retained electrodes (Perrin et al., 1989) on the
      baseline-corrected microvolt signal, then z-scored per channel like every epoch.
  trials_{1,4,16}: each test observation is re-averaged from a random subset of n of
      its single trials (all trials when fewer exist), so the observation noise is the
      recorded trial-to-trial variability rather than synthetic noise. Training uses the
      full averages, as for every other condition.
Training features always come from the clean training fold; test features are
recomputed from the corrupted test signal (the reservoir is re-run on corrupted drives
and projected on the train-only PCA-64 basis; a removed electrode is a silent drive,
whose spike-count block is exactly zero). Protocol and inference as in jcore.py.
Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import perturbations as PT  # noqa: E402
import spline as SP  # noqa: E402
import tcrzem_data as TD  # noqa: E402

REMOVAL = [0.1, 0.3, 0.5]
TRIALS = [1, 4, 16]
CONDS = (["clean"] + [f"remove_{f:.1f}" for f in REMOVAL] + [f"spline_{f:.1f}" for f in REMOVAL]
         + ["amp_5dB", "jitter_50ms"] + [f"trials_{n}" for n in TRIALS])
NAMES = ("Band-power", "ERP-window", "Reservoir")


def trial_average(T, index, te_obs, n, rng):
    """Re-average each test observation from n of its trials (microvolts, not z-scored)."""
    out = np.empty((len(te_obs), T.shape[1], T.shape[2]))
    for j, o in enumerate(te_obs):
        idx = index[o]
        if len(idx) > n:
            idx = np.sort(rng.choice(idx, n, replace=False))
        out[j] = np.asarray(T[idx], dtype=np.float64).mean(axis=0)
    return out


def main():
    t0 = time.time()
    Tr, t_subj, t_val, _ = TD.load_trials()
    subjects = np.unique(t_subj)
    Xuv, y, g = TD.average(Tr, t_subj, t_val, subjects, zscore=False)
    X = TD.zscore_epochs(Xuv)
    N, T, n_ch = X.shape
    classes = np.unique(y)
    # trial index per observation (same order as TD.average)
    order = {}
    for i, (s, v) in enumerate(zip(t_subj, t_val)):
        order.setdefault((s, int(v)), []).append(i)
    obs_trials = [np.array(order[(s, int(v))]) for s, v in zip(g, y)]
    names_, thph = TD.montage()
    pos = SP.sph_to_cart(thph)
    band, erp = J.bandpower(X), J.erp_windows(X)
    B, _ = J.reservoir_codes(X)
    D = B.shape[2]
    print(f"[J2] clean features ready ({time.time() - t0:.0f}s)", flush=True)
    acc = {(n, c): np.zeros((N, classes.size)) for n in NAMES for c in CONDS}
    spline_err = {f"{f:.1f}": [] for f in REMOVAL}
    for seed, fold, tr, te in J.folds(y, g):
        pca, E, v0 = J.pca_embed(B, tr)
        Xtr = {"Band-power": band[tr].reshape(len(tr), -1), "ERP-window": erp[tr].reshape(len(tr), -1),
               "Reservoir": E[tr].reshape(len(tr), -1)}
        models = {n: J.fit_pred(Xtr[n], y[tr], Xtr[n][:1], seed, return_model=True)[1:] for n in NAMES}
        for cname in CONDS:
            Bte = None
            if cname == "clean":
                Xte = X[te]; Bte = B[te]
            elif cname.startswith("remove_"):
                Xte, drop = PT.remove_channels(X[te], float(cname.split("_")[1]), seed, fold)
                Bte = B[te].copy(); Bte[:, drop, :] = 0.0
            elif cname.startswith("spline_"):
                drop = PT.removal_draw(n_ch, float(cname.split("_")[1]), seed, fold)
                rep = SP.repair(Xuv[te], pos, drop)
                Xte = TD.zscore_epochs(rep)
                # repair fidelity on the missing electrodes (z-scored correlation with truth)
                a = Xte[:, :, drop].transpose(0, 2, 1).reshape(-1, T)
                b = X[te][:, :, drop].transpose(0, 2, 1).reshape(-1, T)
                spline_err[cname.split("_")[1]].append(float(np.mean(
                    np.sum(a * b, axis=1) / T)))
            elif cname.startswith("trials_"):
                n = int(cname.split("_")[1])
                rng = np.random.default_rng(30_000 + 1000 * seed + fold + 7 * n)
                Xte = TD.zscore_epochs(trial_average(Tr, obs_trials, te, n, rng))
            else:
                Xte = PT.SIGNAL_CONDITIONS[cname](X[te], seed, fold)
            if Bte is None:
                Bte, _ = J.reservoir_codes(Xte)
            te_feat = {"Band-power": J.bandpower(Xte).reshape(len(te), -1),
                       "ERP-window": J.erp_windows(Xte).reshape(len(te), -1),
                       "Reservoir": pca.transform(Bte.reshape(-1, D)).reshape(len(te), -1)}
            for n in NAMES:
                sc, clf = models[n]
                acc[(n, cname)][te] += clf.predict_proba(sc.transform(te_feat[n]))
        print(f"[J2] seed {seed} fold {fold} ({time.time() - t0:.0f}s)", flush=True)
    preds = {k: J.argmax_pred(v, classes) for k, v in acc.items()}
    res = {"protocol": "signal-level conditions (perturbations.py draws; spline repair on microvolt "
                       "signal then per-epoch z-score; trial-subset re-averaging) applied to test "
                       "subjects only; features recomputed from the corrupted test signal; reservoir "
                       "rho=0.9 re-run, train-only PCA-64; StratifiedGroupKFold(5) x seeds 42-46; "
                       "train-only StandardScaler; balanced L2 logreg; OOF proba averaged over "
                       "partitions; subject-level bootstrap n_boot=%d" % J.N_BOOT,
           "trials_per_observation": {"median": float(np.median([len(o) for o in obs_trials])),
                                      "min": int(min(len(o) for o in obs_trials)),
                                      "max": int(max(len(o) for o in obs_trials))},
           "spline_mean_correlation_missing_channels": {k: float(np.mean(v)) for k, v in spline_err.items()},
           "BA": {}, "paired_ERPwindow_minus_Reservoir": {}, "paired_clean_minus_condition": {},
           "paired_spline_minus_zero": {}}
    for c in CONDS:
        pm = {n: preds[(n, c)] for n in NAMES}
        pt, ci, _ = J.boot_ci(y, g, pm)
        res["BA"][c] = {n: {"BA": pt[n], "ci95": ci[n]} for n in NAMES}
        res["paired_ERPwindow_minus_Reservoir"][c] = J.paired_ci(y, g, pm["ERP-window"], pm["Reservoir"])
        if c != "clean":
            res["paired_clean_minus_condition"][c] = {
                n: J.paired_ci(y, g, preds[(n, "clean")], pm[n]) for n in NAMES}
        print(f"[J2 {c}] " + " | ".join(f"{n}={pt[n]:.3f}[{ci[n][0]:.3f},{ci[n][1]:.3f}]" for n in NAMES),
              flush=True)
    for f in REMOVAL:
        k = f"{f:.1f}"
        res["paired_spline_minus_zero"][k] = {
            n: J.paired_ci(y, g, preds[(n, f"spline_{k}")], preds[(n, f"remove_{k}")]) for n in NAMES}
    J.dump(res, "j2_tcrzem_signal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
