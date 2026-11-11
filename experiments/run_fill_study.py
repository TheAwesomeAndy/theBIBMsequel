#!/usr/bin/env python3
"""Phase 2. Fill-rule study on any condition-averaged dataset in the common format.

For each dataset (cached ERPSet under $SEQUEL_DATA/cache) and each fixed encoder:
    clean
    feature-level fills at each dropout rate: zero, mean, kNN, spatial, spline*
    signal-level removal (zero traces) and signal-level spline repair*
    readout refit on the retained channels
(* needs electrode positions). Reports subject-level BA with CIs, fill contrasts against
zero fill, encoder contrasts, and the Proposition-1 quantities for every zero-fill cell.

    SEQUEL_DATA=/path python experiments/run_fill_study.py erpcore_P3 tcrzem --rates 0.1 0.3 0.5
    python experiments/run_fill_study.py --synthetic --quick          # code check only
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import (TheoryLog, base_parser, cv_settings, removal_draw, protocol,  # noqa: E402
                    write)
from sequel import config as C, datasets, features, fills, reservoir  # noqa: E402


def encoders(win):
    res = reservoir.LIFReservoir()
    return {
        "BSC6": (lambda X: res.encode(X, win, 6).counts, True),
        "Amp6": (lambda X: features.amplitude_bins(X, win, 6), False),
        "ERP-window": (lambda X, t=None: None, False),            # filled in per dataset (needs times)
        "Band-power": (lambda X: features.bandpower(X), False),
    }


def fold_fn_factory(name, fn, use_pca, X, B, positions, rates):
    n_ch = B.shape[1]

    def fn_fold(seed, fold, tr, te):
        if use_pca:
            P = features.BlockPCA().fit(B[tr])
            E, v0, emb = P.transform(B), P.silent_channel(), P.transform
        else:
            E, v0, emb = B.astype(np.float64), np.zeros(B.shape[2]), (lambda b: b.astype(np.float64))
        Xtr = E[tr].reshape(len(tr), -1)
        conds = {"clean": (Xtr, E[te].reshape(len(te), -1), None)}
        for frac in rates:
            pct = int(round(frac * 100))
            drop = removal_draw(n_ch, frac, seed, fold)
            meta = {"drop": drop, "n_feat": E.shape[2], "Xtr": Xtr}
            for rule in ("zero", "mean", "knn", "spatial", "spline"):
                if rule == "spline" and (positions is None or use_pca):
                    continue
                Ete = fills.fill_blocks(E[te], E[tr], drop, rule, positions)
                conds[f"fc{pct}_{rule}"] = (Xtr, Ete.reshape(len(te), -1), meta)
            Ete = E[te].copy()
            Ete[:, drop, :] = v0
            conds[f"sig{pct}_zero"] = (Xtr, Ete.reshape(len(te), -1), None)
            if positions is not None:
                Xrep = fills.interpolate_signal(X[te], drop, positions)
                conds[f"sig{pct}_spline"] = (Xtr, emb(fn(Xrep)).reshape(len(te), -1), None)
            keep = np.setdiff1d(np.arange(n_ch), drop)
            conds[f"refit{pct}"] = (E[tr][:, keep].reshape(len(tr), -1),
                                    E[te][:, keep].reshape(len(te), -1), None)
        return conds

    return fn_fold


def run_dataset(ep, positions, splits, n_boot, rates, window_ms=C.RES_WINDOWS_REL_MS["post"]):
    y, g = ep.y, ep.groups
    win = C.window_samples(window_ms, onset=ep.onset)
    enc = encoders(win)
    enc["ERP-window"] = (lambda X: features.erp_windows(X, ep.times_ms), False)
    enc["LPP-300-1000"] = (lambda X: features.lpp_window(X, ep.times_ms), False)
    preds, th = {}, {}
    for name, (fn, pca) in enc.items():
        t0 = time.time()
        B = fn(ep.X)
        log = TheoryLog()
        preds[name] = protocol.pooled_oof(fold_fn_factory(name, fn, pca, ep.X, B, positions, rates),
                                          y, splits, hook=log)
        th[name] = log.summary()
        print(f"  [{name}] clean BA={np.mean(preds[name]['clean'] == y):.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    out = {"chance": 1.0 / len(np.unique(y)), "window_samples": list(win), "encoders": {}}
    for name, pm in preds.items():
        ba = protocol.bootstrap_ba(y, g, pm, n_boot)
        con = {}
        for frac in rates:
            pct = int(round(frac * 100))
            for c in [k for k in pm if k.startswith((f"fc{pct}_", f"sig{pct}_", f"refit{pct}"))]:
                if c != f"fc{pct}_zero":
                    con[f"{c}_minus_fc{pct}_zero"] = protocol.paired_diff(y, g, pm[c], pm[f"fc{pct}_zero"], n_boot)
        out["encoders"][name] = {"BA": ba, "contrasts": con, "theory": th[name]}
    out["vs_ERP-window"] = {n: {c: protocol.paired_diff(y, g, preds[n][c], preds["ERP-window"][c], n_boot)
                                for c in ("clean", f"fc{int(rates[len(rates) // 2] * 100)}_mean")}
                            for n in preds if n != "ERP-window"}
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("names", nargs="*", default=[])
    p.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--decimation", default="fir")
    p.add_argument("--norm", default="baseline_z")
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    res = {"design": vars(args) | {"cv_seeds": list(seeds), "n_boot": n_boot}, "datasets": {}}
    if args.synthetic:
        from tests.synthetic import electrode_positions, raw_epochs
        raw, y, g = raw_epochs(n_subj=24)
        sets = {"synthetic": datasets.ERPSet("synthetic", raw, y, g, 1024, 205,
                                             [f"c{i}" for i in range(34)], ["a", "b", "c"],
                                             electrode_positions(34))}
    else:
        sets = {n: datasets.cached(n) for n in args.names}
    for name, s in sets.items():
        ep = s.to_epochs(args.decimation, args.norm)
        splits = protocol.make_splits(ep.y, ep.groups, seeds)
        print(f"[{name}] {s.summary()}", flush=True)
        res["datasets"][name] = {"summary": s.summary(),
                                 "results": run_dataset(ep, s.positions, splits, n_boot, args.rates)}
    write(res, args.out or "p2_fill_study.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
