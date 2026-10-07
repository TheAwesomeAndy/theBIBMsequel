#!/usr/bin/env python3
"""J4: the operating point across reservoir draws (external TCRZEM cohort).

The conference sweep used one weight draw. Here the identical reservoir (Xavier-uniform
W_in and W, W rescaled to spectral radius rho, leak 0.05, threshold 0.5, 256 units) is
drawn with ten seeds (42-51), and for every draw:

1. Label-free order parameter: damage spreading on the cohort's own electrode drives
   (conference definition: one spike flipped at t = 10, normalized Hamming distance
   averaged over steps 21..69 and 400 sampled drives) on rho = 0, 0.1, ..., 2.0; the
   transition rho* is the half-maximum crossing, linearly interpolated. Mean firing rate
   in the analysis window is recorded on the same grid.
2. Accuracy: subject-level BA (train-only PCA-64, conference protocol) at
   rho in {0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5}, clean and with 30% of the
   electrodes removed at the signal level (silent drive).
3. Edge maps (draw 42): rho* as a function of threshold (0.25, 0.5, 1.0) and leak
   (0.02, 0.05, 0.1, 0.2), and clean BA on the conference rho grid for each threshold,
   to test whether the accuracy optimum follows the measured transition.

Paired subject-level contrasts against rho = 0.9 are computed within each draw.
The order parameter always uses the conference definition on the conference-matched
drives (-200..+800 ms, z-scored; flip at t = 10, window ends at t = 70). The accuracy
sweep uses the epoch selected by $TCRZEM_EPOCH (standard, or the pre-specified long
epoch, whose BSC6 window covers 0..+2484 ms). Partial results are written after every
draw. Writes aggregate JSON only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import tcrzem_data as TD  # noqa: E402

DRAWS = list(range(42, 52))
RHO_FINE = [round(r, 2) for r in np.arange(0.0, 2.01, 0.1)]
RHO_ACC = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]
RHO_CONF = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
THETAS = [0.25, 0.5, 1.0]
BETAS = [0.02, 0.05, 0.1, 0.2]


def accuracy(X, y, g, F, classes, **kw):
    """Clean and 30%-removal OOF predictions for one reservoir configuration."""
    N, _, n_ch = X.shape
    B, rate = J.reservoir_codes(X, **kw)
    acc = {"clean": np.zeros((N, classes.size)), "signal30": np.zeros((N, classes.size))}
    for seed, fold, tr, te in F:
        pca, E, v0 = J.pca_embed(B, tr)
        Z = E.reshape(N, -1)
        _, sc, clf = J.fit_pred(Z[tr], y[tr], Z[:1], seed, return_model=True)
        acc["clean"][te] += clf.predict_proba(sc.transform(Z[te]))
        Ete = E[te].copy()
        Ete[:, J.removal_draw(n_ch, 0.3, seed, fold), :] = v0
        acc["signal30"][te] += clf.predict_proba(sc.transform(Ete.reshape(len(te), -1)))
    return {k: J.argmax_pred(v, classes) for k, v in acc.items()}, rate


def standard_drives():
    """Conference-matched drives in either epoch mode: first 256 samples, z-scored per epoch."""
    Xuv, _, _ = TD.load_cohort(zscore=False)
    return TD.zscore_epochs(Xuv[:, :256, :])


def dmg(Xs, rho, **kw):
    return J.damage(Xs, rho=rho, t_start=10, t_end=70, **kw)


def main():
    t0 = time.time()
    X, y, g = TD.load_cohort()
    Xs = standard_drives()
    classes = np.unique(y)
    F = J.folds(y, g)
    res = {"epoch": J.EPOCH, "protocol": "reservoir draws 42-51 (Xavier-uniform, W rescaled to rho), leak 0.05, threshold "
                       "0.5, 256 units; damage spreading on cohort drives (flip at t=10, steps 21-69, 400 "
                       "drives); rho* = half-maximum crossing (linear interpolation) on rho 0..2 step 0.1; "
                       "accuracy with train-only PCA-64, StratifiedGroupKFold(5) x seeds 42-46, balanced "
                       "L2 logreg, subject-level bootstrap n_boot=%d" % J.N_BOOT,
           "rho_fine": RHO_FINE, "rho_acc": RHO_ACC, "draws": {}}
    for d in DRAWS:
        r = {"damage": {}, "rate": {}, "BA": {}, "paired_vs_rho0.9": {}}
        for rho in RHO_FINE:
            r["damage"][str(rho)] = dmg(Xs, rho, seed=d)
        r["rho_star"] = J.rho_star(RHO_FINE, [r["damage"][str(x)] for x in RHO_FINE])
        preds = {}
        for rho in RHO_ACC:
            preds[rho], r["rate"][str(rho)] = accuracy(X, y, g, F, classes, rho=rho, seed=d)
            r["BA"][str(rho)] = {k: J.ba_entry(y, g, p) for k, p in preds[rho].items()}
        for rho in RHO_ACC:
            if rho != 0.9:
                r["paired_vs_rho0.9"][str(rho)] = {k: J.paired_ci(y, g, preds[rho][k], preds[0.9][k])
                                                   for k in ("clean", "signal30")}
        res["draws"][str(d)] = r
        best = max(RHO_ACC, key=lambda x: r["BA"][str(x)]["clean"]["BA"])
        print(f"[J4 draw {d}] rho*={r['rho_star']:.3f} best clean rho={best} "
              + " ".join(f"{x}:{r['BA'][str(x)]['clean']['BA']:.3f}" for x in RHO_ACC)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        J.dump(res, "j4_tcrzem_edge.json")
    # summary over draws
    S = {"rho_star": [res["draws"][str(d)]["rho_star"] for d in DRAWS]}
    for cond in ("clean", "signal30"):
        M = np.array([[res["draws"][str(d)]["BA"][str(x)][cond]["BA"] for x in RHO_ACC] for d in DRAWS])
        S[f"BA_{cond}_mean"] = M.mean(0).tolist()
        S[f"BA_{cond}_sd"] = M.std(0, ddof=1).tolist()
        S[f"BA_{cond}_min"] = M.min(0).tolist()
        S[f"BA_{cond}_max"] = M.max(0).tolist()
        S[f"argmax_rho_{cond}"] = [RHO_ACC[int(i)] for i in M.argmax(1)]
        S[f"n_draws_rho0_below_rho0.9_ci_excludes0_{cond}"] = int(sum(
            res["draws"][str(d)]["paired_vs_rho0.9"]["0.0"][cond]["ci95"][1] < 0 for d in DRAWS))
        S[f"n_draws_rho1.5_below_rho0.9_ci_excludes0_{cond}"] = int(sum(
            res["draws"][str(d)]["paired_vs_rho0.9"]["1.5"][cond]["ci95"][1] < 0 for d in DRAWS))
    Dm = np.array([[res["draws"][str(d)]["damage"][str(x)] for x in RHO_FINE] for d in DRAWS])
    S["damage_mean"] = Dm.mean(0).tolist(); S["damage_sd"] = Dm.std(0, ddof=1).tolist()
    Rm = np.array([[res["draws"][str(d)]["rate"][str(x)] for x in RHO_ACC] for d in DRAWS])
    S["rate_mean"] = Rm.mean(0).tolist()
    res["summary"] = S
    print(f"[J4] rho* over draws: mean {np.mean(S['rho_star']):.3f} sd {np.std(S['rho_star'], ddof=1):.3f} "
          f"range [{min(S['rho_star']):.3f},{max(S['rho_star']):.3f}]", flush=True)
    J.dump(res, "j4_tcrzem_edge.json")
    # edge maps for draw 42
    em = {"theta": {}, "beta": {}}
    for th in THETAS:
        dm = [dmg(Xs, rho, theta=th) for rho in RHO_FINE]
        ent = {"damage": dict(zip(map(str, RHO_FINE), dm)), "rho_star": J.rho_star(RHO_FINE, dm), "BA_clean": {}}
        for rho in RHO_CONF:
            if th == J.THETA:                       # identical to draw 42 of the main sweep
                ent["BA_clean"][str(rho)] = res["draws"]["42"]["BA"][str(rho)]["clean"]
                ent.setdefault("rate", {})[str(rho)] = res["draws"]["42"]["rate"][str(rho)]
                continue
            p, rate = accuracy(X, y, g, F, classes, rho=rho, theta=th)
            ent["BA_clean"][str(rho)] = J.ba_entry(y, g, p["clean"])
            ent.setdefault("rate", {})[str(rho)] = rate
        em["theta"][str(th)] = ent
        print(f"[J4 theta={th}] rho*={ent['rho_star']:.3f} "
              + " ".join(f"{x}:{ent['BA_clean'][str(x)]['BA']:.3f}" for x in RHO_CONF), flush=True)
    for be in BETAS:
        dm = [dmg(Xs, rho, beta=be) for rho in RHO_FINE]
        em["beta"][str(be)] = {"damage": dict(zip(map(str, RHO_FINE), dm)), "rho_star": J.rho_star(RHO_FINE, dm)}
        print(f"[J4 beta={be}] rho*={em['beta'][str(be)]['rho_star']:.3f}", flush=True)
    res["edge_maps_draw42"] = em
    J.dump(res, "j4_tcrzem_edge.json")
    print(f"[J4] total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
