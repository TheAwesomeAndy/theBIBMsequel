#!/usr/bin/env python3
"""Phase 1c. Within-subject label permutation with the headline estimator.

BIBM ran 20 permutations. A journal reviewer will expect at least 1000, with the same
readout as the reported accuracy. Each encoder's fold features are computed once (they are
label-free) and reused for every permutation.

    python experiments/run_permutation.py --shape-dir /path/batch_data --n-perm 1000 --jobs 4
"""
from __future__ import annotations

import sys

from common import base_parser, cv_settings, load_raw, protocol, write  # noqa: E402
from sequel import config as C, data, features, reservoir  # noqa: E402


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--decimation", default="fir")
    p.add_argument("--norm", default="baseline_z")
    p.add_argument("--window", default="post", choices=list(C.RES_WINDOWS_REL_MS))
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--encoders", nargs="+", default=["BSC6", "ERP-window", "Band-power", "Amp6"])
    args = p.parse_args()
    seeds, _ = cv_settings(args)
    raw, y, g = load_raw(args)
    ep = data.preprocess(raw, y, g, args.decimation, args.norm)
    win = C.window_samples(C.RES_WINDOWS_REL_MS[args.window], onset=ep.onset)
    splits = protocol.make_splits(y, g, seeds)
    out = {"design": vars(args) | {"window_samples": list(win), "cv_seeds": list(seeds)},
           "encoders": {}}
    for name in args.encoders:
        if name == "BSC6":
            B = reservoir.LIFReservoir().encode(ep.X, win, 6).counts
            cache = []
            for s, f, tr, te in splits:
                P = features.BlockPCA().fit(B[tr])
                E = P.transform(B).reshape(len(y), -1)
                cache.append((s, f, tr, te, E[tr], E[te]))
        else:
            B = {"ERP-window": lambda: features.erp_windows(ep.X, ep.times_ms),
                 "Band-power": lambda: features.bandpower(ep.X),
                 "Amp6": lambda: features.amplitude_bins(ep.X, win, 6)}[name]()
            F = B.reshape(len(y), -1)
            cache = [(s, f, tr, te, F[tr], F[te]) for s, f, tr, te in splits]
        out["encoders"][name] = protocol.permutation_test(cache, y, g, args.n_perm, n_jobs=args.jobs)
        r = out["encoders"][name]
        print(f"[{name}] BA={r['observed_BA']:.3f} null={r['null_mean']:.3f}+-{r['null_sd']:.3f} "
              f"max={r['null_max']:.3f} p={r['p']:.4f}", flush=True)
    write(out, args.out or "p1c_permutation.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
