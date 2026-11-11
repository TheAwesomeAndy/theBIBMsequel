#!/usr/bin/env python3
"""Automated data check of the SHAPE export (the check recommended by the data provider).

Per file: dimensions (expected 1229 x 34), non-finite values, flat channels, baseline mean
(the export is baseline-corrected over the first 205 samples), amplitude range. Across
files: complete Neg/Neu/Pos panels per subject and exact duplicate files. Writes aggregate
counts only (subject IDs are listed for anomalies so they can be reported back to the lab).

    python experiments/check_shape_files.py --shape-dir /path/batch_data
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import base_parser, write  # noqa: E402

FNAME = re.compile(r"SHAPE_Community_(?P<sid>[^_]+)_IAPS(?P<cond>Neg|Neu|Pos)_BC\.txt$")


def main() -> int:
    args = base_parser(__doc__).parse_args()
    d = Path(args.shape_dir)
    files = sorted(p for p in d.iterdir() if p.is_file())
    bad_name, bad_shape, nonfinite, flat, big_baseline, panels = [], [], [], [], [], defaultdict(set)
    hashes = defaultdict(list)
    amp = []
    for p in files:
        m = FNAME.match(p.name)
        if not m:
            bad_name.append(p.name)
            continue
        panels[m["sid"]].add(m["cond"])
        hashes[hashlib.sha1(p.read_bytes()).hexdigest()].append(p.name)
        x = np.loadtxt(p)
        if x.shape != (1229, 34):
            bad_shape.append([p.name, list(x.shape)])
            continue
        if not np.isfinite(x).all():
            nonfinite.append(p.name)
        sd = x.std(axis=0)
        if (sd < 1e-6).any():
            flat.append([p.name, np.where(sd < 1e-6)[0].tolist()])
        if np.abs(x[:205].mean(axis=0)).max() > 0.01:
            big_baseline.append(p.name)
        amp.append(np.abs(x).max())
    incomplete = {s: sorted(c) for s, c in panels.items() if c != {"Neg", "Neu", "Pos"}}
    dups = [v for v in hashes.values() if len(v) > 1]
    out = {"n_files": len(files), "n_subjects": len(panels),
           "expected_shape": [1229, 34], "wrong_shape": bad_shape, "unrecognized_names": bad_name,
           "nonfinite_files": nonfinite, "flat_channel_files": flat,
           "baseline_mean_over_0.01uV": big_baseline, "incomplete_panels": incomplete,
           "duplicate_file_groups": dups,
           "max_abs_amplitude_uV": {"median": float(np.median(amp)), "p99": float(np.percentile(amp, 99)),
                                    "max": float(np.max(amp))}}
    for k, v in out.items():
        print(f"{k}: {v}")
    write(out, args.out or "shape_file_check.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
