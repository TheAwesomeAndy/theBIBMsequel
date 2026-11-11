#!/usr/bin/env python3
"""Phase 1b. Spectral-radius sweep over many reservoir weight draws, with the missing
threshold-density measurement.

BIBM used one weight draw (seed 42) and left the one-step branching threshold
rho_c ~ 1 / (p_theta * kappa_bar) undetermined because p_theta (the density of
pre-threshold membrane values at theta) was not measured. For every (weight seed, rho)
this script records:

    firing rate, damage-spreading order parameter (raw and rate-normalized),
    p_theta, kappa_bar, branching count m(rho) = rho * p_theta * kappa_bar,
    and clean subject-level BA of BSC6 / train-only PCA-64.

It then reports, per seed, the half-maximum damage crossing rho* (linear interpolation, as
in BIBM), the first rho with m >= 1 (interpolated), and the across-seed spread of BA at
each rho. The rho = 0 versus rho = 0.9 contrast is reported per seed and pooled.

    python experiments/run_reservoir_dynamics.py --shape-dir /path/batch_data
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import base_parser, cv_settings, encoder_fold_fn, load_raw, protocol, write  # noqa: E402
from sequel import config as C, data, reservoir  # noqa: E402


def crossing(xs, vs, level):
    """First x where the piecewise-linear curve vs(x) reaches ``level`` (None if never)."""
    for i in range(1, len(xs)):
        if vs[i - 1] < level <= vs[i]:
            return float(xs[i - 1] + (level - vs[i - 1]) * (xs[i] - xs[i - 1]) / (vs[i] - vs[i - 1]))
    return None


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--decimation", default="fir")
    p.add_argument("--norm", default="baseline_z")
    p.add_argument("--window", default="post", choices=list(C.RES_WINDOWS_REL_MS))
    p.add_argument("--weight-seeds", type=int, nargs="+", default=[42] + list(range(1, 10)))
    p.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    p.add_argument("--rho-fine", type=float, nargs="+",
                   default=[round(r, 1) for r in np.arange(0.0, 2.01, 0.1)])
    p.add_argument("--n-drive", type=int, default=400)
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    raw, y, g = load_raw(args)
    ep = data.preprocess(raw, y, g, args.decimation, args.norm)
    win = C.window_samples(C.RES_WINDOWS_REL_MS[args.window], onset=ep.onset)
    splits = protocol.make_splits(y, g, seeds)
    out = {"design": vars(args) | {"window_samples": list(win), "cv_seeds": list(seeds),
                                   "n_boot": n_boot}, "seeds": {}}
    preds_by = {}
    for ws in args.weight_seeds:
        rec = {"dynamics": {}, "BA": {}}
        for rho in args.rho_fine:
            res = reservoir.LIFReservoir(rho=rho, weight_seed=ws)
            dmg = res.damage(ep.X, win, n_drive=args.n_drive)
            mem = res.encode(ep.X[: max(1, len(y) // 10)], win, 6, membrane_stats=True)
            kb = res.kappa_bar()
            rec["dynamics"][str(rho)] = dmg | {"p_theta": mem.p_theta, "kappa_bar": kb,
                                               "m": reservoir.branching_count(rho, mem.p_theta, kb)}
        for rho in args.rho:
            t0 = time.time()
            run = reservoir.LIFReservoir(rho=rho, weight_seed=ws).encode(ep.X, win, 6)
            pm = protocol.pooled_oof(encoder_fold_fn(run.counts, True, drop_frac=0.0), y, splits)
            preds_by[(ws, rho)] = pm["clean"]
            rec["BA"][str(rho)] = protocol.bootstrap_ba(y, g, pm, n_boot)["clean"] | {"rate": run.rate}
            print(f"[seed {ws} rho {rho}] BA={rec['BA'][str(rho)]['BA']:.3f} rate={run.rate:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        xs = args.rho_fine
        dm = [rec["dynamics"][str(r)]["damage"] for r in xs]
        mm = [rec["dynamics"][str(r)]["m"] for r in xs]
        rec["rho_star_half_max_damage"] = crossing(xs, dm, 0.5 * max(dm)) if max(dm) > 0 else None
        rec["rho_c_branching_m_eq_1"] = crossing(xs, mm, 1.0)
        if 0.0 in args.rho and 0.9 in args.rho:
            rec["rho0_minus_rho0.9"] = protocol.paired_diff(y, g, preds_by[(ws, 0.0)],
                                                           preds_by[(ws, 0.9)], n_boot)
        out["seeds"][str(ws)] = rec
    out["across_seeds"] = {str(rho): {
        "BA_mean": float(np.mean([out["seeds"][str(s)]["BA"][str(rho)]["BA"] for s in args.weight_seeds])),
        "BA_sd": float(np.std([out["seeds"][str(s)]["BA"][str(rho)]["BA"] for s in args.weight_seeds], ddof=1))
        if len(args.weight_seeds) > 1 else None} for rho in args.rho}
    stars = [out["seeds"][str(s)]["rho_star_half_max_damage"] for s in args.weight_seeds]
    rcs = [out["seeds"][str(s)]["rho_c_branching_m_eq_1"] for s in args.weight_seeds]
    out["across_seeds"]["rho_star"] = [s for s in stars]
    out["across_seeds"]["rho_c"] = [r for r in rcs]
    write(out, args.out or "p1b_reservoir_dynamics.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
