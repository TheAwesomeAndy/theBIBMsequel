#!/usr/bin/env python3
"""Reproduce BIBM camera-ready numbers with the new package before changing anything.

Runs on the BIBM pickle (legacy signal, legacy windows, rho = 0.9, train-only PCA) and
compares point balanced accuracies against bibm_baseline/outputs/aggregate/paper_values.csv.
Exit status 1 on any mismatch above --tol. This is the gate that every later result is
built on: if it fails, the new code does not implement the BIBM protocol.

    python experiments/check_bibm_regression.py --pickle /path/shape_features_211.pkl
"""
from __future__ import annotations

import csv
import sys

from sklearn.metrics import balanced_accuracy_score

from common import ROOT, base_parser, encoder_fold_fn, protocol, write  # noqa: E402
from sequel import config as C, data, features, reservoir  # noqa: E402

EXPECTED = {  # (table, row, column) in paper_values.csv -> (encoder, condition)
    ("tab:clean", "Band-power", "BA"): ("Band-power", "clean"),
    ("tab:clean", "ERP-window", "BA"): ("ERP-window", "clean"),
    ("tab:clean", "Reservoir", "BA"): ("Reservoir", "clean"),
    ("tab:dropconf", "zero", "Band-power"): ("Band-power", "fc30_zero"),
    ("tab:dropconf", "zero", "ERP-window"): ("ERP-window", "fc30_zero"),
    ("tab:dropconf", "zero", "Reservoir"): ("Reservoir", "fc30_zero"),
    ("tab:dropconf", "mean", "Band-power"): ("Band-power", "fc30_mean"),
    ("tab:dropconf", "mean", "ERP-window"): ("ERP-window", "fc30_mean"),
    ("tab:dropconf", "mean", "Reservoir"): ("Reservoir", "fc30_mean"),
    ("tab:dropconf", "knn", "ERP-window"): ("ERP-window", "fc30_knn"),
    ("tab:dropconf", "spatial", "ERP-window"): ("ERP-window", "fc30_spatial"),
    ("tab:rawsignal", "remove_0.3", "Reservoir"): ("Reservoir", "sig30"),
}


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()
    ep = data.load_bibm_pickle(args.pickle)
    y, g = ep.y, ep.groups
    blocks = {
        "Band-power": (features.bandpower(ep.X), False),
        "ERP-window": (features.erp_windows_legacy(ep.X), False),
        "Reservoir": (reservoir.LIFReservoir().encode(ep.X, C.LEGACY_RES_WINDOW).counts, True),
    }
    splits = protocol.make_splits(y, g)
    got = {}
    for name, (B, pca) in blocks.items():
        preds = protocol.pooled_oof(encoder_fold_fn(B, pca, 0.3, refit=False), y, splits)
        got[name] = {k: float(balanced_accuracy_score(y, v)) for k, v in preds.items()}
        print(f"[{name}] " + " ".join(f"{k}={v:.6f}" for k, v in got[name].items()), flush=True)

    ref = {}
    with open(ROOT / "bibm_baseline" / "outputs" / "aggregate" / "paper_values.csv") as fh:
        for r in csv.DictReader(fh):
            ref[(r["table"], r["row"], r["column"])] = float(r["value"])
    rows, bad = [], 0
    for key, (enc, cond) in EXPECTED.items():
        exp, obs = ref[key], got[enc][cond]
        ok = abs(exp - obs) <= args.tol
        bad += not ok
        rows.append({"paper_cell": list(key), "expected": exp, "reproduced": obs, "ok": ok})
        print(f"{'OK ' if ok else 'BAD'} {key}: paper {exp:.6f} new {obs:.6f}")
    write({"check": "BIBM regression", "tol": args.tol, "rows": rows, "n_fail": bad},
          args.out or "bibm_regression.json", args)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
