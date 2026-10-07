#!/usr/bin/env python3
"""Camera-ready experiment E4: signal-level electrode removal (10/30/50%), amplitude noise and
temporal jitter for the fixed encoders (band-power, ERP-window, reservoir), matching E3 (EEGNet)
exactly. Supersedes experiment2_rawsignal.py for Table III: train and test reservoir codes are
both embedded with PCA.transform (experiment2 used fit_transform for training codes, which under
the randomized SVD solver differs slightly from transform).

Every perturbation is applied to the analysed test signal BEFORE feature extraction, using
the shared draws in perturbations.py (so EEGNet in E3 sees the same corruption). Training
features come from the clean training fold; test features are recomputed from the corrupted
test signal:
  - band-power: Welch estimator of experiment2_rawsignal.py on the corrupted signal;
  - ERP-window: window means of the corrupted signal;
  - reservoir (rho = 0.9): the LIF reservoir is re-run on the corrupted test drives, and the
    BSC6 code is projected on the PCA basis fitted to the clean training subjects only.
Protocol and inference are the source ones (StratifiedGroupKFold(5) x seeds 42-46, train-only
standardization, balanced L2 logistic, OOF probabilities averaged over partitions,
subject-level bootstrap). Writes aggregate JSON only.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment2_rawsignal as E2S  # noqa: E402
import experiment_camera_ready as EC  # noqa: E402
import perturbations as PT  # noqa: E402
import reanalysis_subject_bootstrap as RB  # noqa: E402

RHO = 0.9
REMOVAL = [0.1, 0.3, 0.5]
CONDS = ["clean"] + [f"remove_{f:.1f}" for f in REMOVAL] + list(PT.SIGNAL_CONDITIONS)


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    N, T, C = X.shape; classes = np.unique(y)
    t0 = time.time()
    band = E2S.bandpower_all(X); erp = E2S.erp_all(X)
    B, _ = EC.bsc6(X, RHO)                                   # clean BSC6 (N, 34, 1536)
    print(f"[E4] clean features ready ({time.time() - t0:.0f}s)", flush=True)
    names = ("Band-power", "ERP-window", "Reservoir")
    acc = {(n, c): np.zeros((N, classes.size)) for n in names for c in CONDS}
    for seed in EC.SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((N, 1)), y, groups=g)):
            D = B.shape[2]
            pca = PCA(n_components=EC.PCA_K, random_state=EC.SEED).fit(B[tr].reshape(-1, D))
            Etr = pca.transform(B[tr].reshape(-1, D)).reshape(len(tr), -1)
            Xtr_feat = {"Band-power": band[tr].reshape(len(tr), -1),
                        "ERP-window": erp[tr].reshape(len(tr), -1),
                        "Reservoir": Etr}
            for cname in CONDS:
                if cname == "clean":
                    Xte = X[te]; Bte = B[te]
                elif cname.startswith("remove_"):
                    # a removed electrode is a zero trace; a zero drive from a zero state
                    # produces no spikes, so its BSC6 block is exactly zero (no re-run needed)
                    Xte, drop = PT.remove_channels(X[te], float(cname.split("_")[1]), seed, fold)
                    Bte = B[te].copy(); Bte[:, drop, :] = 0.0
                else:
                    Xte = PT.SIGNAL_CONDITIONS[cname](X[te], seed, fold)
                    Bte, _ = EC.bsc6(Xte, RHO)
                te_feat = {"Band-power": E2S.bandpower_all(Xte).reshape(len(te), -1),
                           "ERP-window": E2S.erp_all(Xte).reshape(len(te), -1),
                           "Reservoir": pca.transform(Bte.reshape(-1, D)).reshape(len(te), -1)}
                for n in names:
                    acc[(n, cname)][te] += EC.fit_pred(Xtr_feat[n], y[tr], te_feat[n], seed)
            print(f"[E4] seed {seed} fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
    preds = {k: classes[np.argmax(v, axis=1)] for k, v in acc.items()}
    res = {"protocol": "signal-level amplitude noise (5 dB) and alignment jitter (+/-50 ms) from "
           "perturbations.py; features recomputed from the corrupted test signal; reservoir "
           "rho=0.9 re-run on corrupted drives, PCA-64 fit on clean training subjects; "
           "StratifiedGroupKFold(5) x seeds 42-46; train-only StandardScaler; balanced L2 logreg; "
           "OOF proba averaged over partitions; subject-level bootstrap n_boot=%d" % RB.N_BOOT,
           "BA": {}, "paired_ERPwindow_minus_Reservoir": {}, "paired_clean_minus_condition": {}}
    for c in CONDS:
        pm = {n: preds[(n, c)] for n in names}
        pt, ci, _ = RB.subject_bootstrap(y, g, pm)
        res["BA"][c] = {n: {"BA": pt[n], "ci95": list(ci[n])} for n in names}
        m, lo, hi, _ = RB.paired_diff_ci(y, g, pm["ERP-window"], pm["Reservoir"])
        res["paired_ERPwindow_minus_Reservoir"][c] = {"mean_diff": m, "ci95": [lo, hi]}
        if c != "clean":
            res["paired_clean_minus_condition"][c] = {}
            for n in names:
                m, lo, hi, _ = RB.paired_diff_ci(y, g, preds[(n, "clean")], pm[n])
                res["paired_clean_minus_condition"][c][n] = {"mean_diff": m, "ci95": [lo, hi]}
        print(f"[E4 {c}] " + " | ".join(f"{n}={pt[n]:.3f}[{ci[n][0]:.3f},{ci[n][1]:.3f}]" for n in names),
              flush=True)
    out = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "e4_signal_perturbations.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[out] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
