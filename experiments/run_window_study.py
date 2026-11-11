#!/usr/bin/env python3
"""Phase 1a. Corrected window and preprocessing for every fixed encoder.

Question: does the BIBM reservoir result survive (i) an anti-aliased signal, (ii)
normalization that keeps post-stimulus signal out of the pre-stimulus samples, and (iii)
reservoir windows that cover the evoked components? And does the reservoir add anything
over a matched non-spiking control with the same bins (Amp6)?

Grid (one aggregate JSON):
    decimation   legacy | fir
    norm         epoch_z | baseline_z
    window       bibm (-160..+74 ms) | early (0..250) | late (250..800) | post (0..800)
Encoders per cell:
    BSC6 (reservoir, 6 bins, train-only PCA-64)   BSC1 (total count, PCA-64)
    Amp6 (mean amplitude in the same 6 bins)      ERP-window (-120..0, 50..250, 250..600 ms)
    Band-power (Welch, five bands)
Conditions per encoder: clean; 30% feature-level fills (zero, mean, kNN, spatial);
30% signal-level removal; readout refit on retained channels.
Contrasts: BSC6 minus each other encoder (clean), mean minus zero fill and refit minus
zero fill per encoder, and Proposition-1 quantities (non-centeredness, fill-bias index).

    python experiments/run_window_study.py --shape-dir /path/batch_data
    python experiments/run_window_study.py --synthetic --quick      # code check only
"""
from __future__ import annotations

import itertools
import sys
import time

from common import (TheoryLog, base_parser, cv_settings, encoder_fold_fn, load_raw,  # noqa: E402
                    protocol, write)
from sequel import config as C, data, features, reservoir  # noqa: E402


def run_cell(ep, window_ms, splits, n_boot, rho=C.RHO, weight_seed=C.WEIGHT_SEED):
    y, g = ep.y, ep.groups
    win = C.window_samples(window_ms, onset=ep.onset)
    run = reservoir.LIFReservoir(rho=rho, weight_seed=weight_seed).encode(ep.X, win, 6)
    enc = {
        "BSC6": (run.counts, True),
        "BSC1": (reservoir.bsc1_from_bsc(run.counts), True),
        "Amp6": (features.amplitude_bins(ep.X, win, 6), False),
        "ERP-window": (features.erp_windows(ep.X, ep.times_ms), False),
        "Band-power": (features.bandpower(ep.X), False),
    }
    preds, theory_out = {}, {}
    for name, (B, pca) in enc.items():
        log = TheoryLog()
        preds[name] = protocol.pooled_oof(encoder_fold_fn(B, pca, 0.3), y, splits, hook=log)
        theory_out[name] = log.summary()
    out = {"window_samples": list(win), "firing_rate": run.rate, "encoders": {}}
    for name, pm in preds.items():
        ba = protocol.bootstrap_ba(y, g, pm, n_boot)
        con = {f"{c}_minus_fc30_zero": protocol.paired_diff(y, g, pm[c], pm["fc30_zero"], n_boot)
               for c in ("fc30_mean", "fc30_knn", "fc30_spatial", "refit30")}
        out["encoders"][name] = {"BA": ba, "fill_contrasts": con, "theory": theory_out[name]}
    out["BSC6_minus"] = {
        other: {c: protocol.paired_diff(y, g, preds["BSC6"][c], preds[other][c], n_boot)
                for c in ("clean", "fc30_mean", "sig30")}
        for other in enc if other != "BSC6"}
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--decimation", nargs="+", default=["legacy", "fir"])
    p.add_argument("--norm", nargs="+", default=["epoch_z", "baseline_z"])
    p.add_argument("--windows", nargs="+", default=list(C.RES_WINDOWS_REL_MS))
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    raw, y, g = load_raw(args)
    splits = protocol.make_splits(y, g, seeds)
    res = {"design": {"decimation": args.decimation, "norm": args.norm, "windows":
                      {w: C.RES_WINDOWS_REL_MS[w] for w in args.windows},
                      "cv_seeds": list(seeds), "n_boot": n_boot, "n_obs": int(len(y)),
                      "n_subjects": int(len(set(g)))}, "cells": {}}
    for dec, norm in itertools.product(args.decimation, args.norm):
        ep = data.preprocess(raw, y, g, dec, norm)
        for w in args.windows:
            t0 = time.time()
            key = f"{dec}|{norm}|{w}"
            res["cells"][key] = run_cell(ep, C.RES_WINDOWS_REL_MS[w], splits, n_boot)
            e = res["cells"][key]["encoders"]
            print(f"[{key}] " + " ".join(f"{n}={e[n]['BA']['clean']['BA']:.3f}" for n in e)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    write(res, args.out or "p1a_window_study.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
