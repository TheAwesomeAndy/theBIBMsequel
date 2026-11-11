#!/usr/bin/env python3
"""Paired clean-accuracy contrasts: BSC6 reservoir minus every other encoder.

The fill study reports each encoder's CI separately; with 35-40 subjects those overlap
even for sizeable differences. This script gives the paired subject-level bootstrap
contrast (same resamples for both encoders) for every dataset and normalization.

    SEQUEL_DATA=/path python experiments/run_encoder_contrasts.py erpcore_P3 erpcore_N400 \
        erpcore_N170 tcrzem --norms baseline_z global_scale
"""
from __future__ import annotations

import sys

from common import base_parser, cv_settings, protocol, write  # noqa: E402
from sequel import config as C, datasets, features, reservoir  # noqa: E402


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--norms", nargs="+", default=["global_scale"])
    p.add_argument("--window", default="post", choices=list(C.RES_WINDOWS_REL_MS))
    p.add_argument("--shape-montage-hypothesis", action="store_true")
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    out = {"design": vars(args) | {"cv_seeds": list(seeds), "n_boot": n_boot}, "results": {}}
    for name in args.names:
        s = (datasets.shape(args.shape_dir, args.shape_montage_hypothesis) if name == "shape"
             else datasets.cached(name))
        for norm in args.norms:
            ep = s.to_epochs("fir", norm)
            win = C.window_samples(C.RES_WINDOWS_REL_MS[args.window], onset=ep.onset)
            enc = {"BSC6": (reservoir.LIFReservoir().encode(ep.X, win, 6).counts, True),
                   "Amp6": (features.amplitude_bins(ep.X, win, 6), False),
                   "LPP-300-1000": (features.lpp_window(ep.X, ep.times_ms), False),
                   "ERP-window": (features.erp_windows(ep.X, ep.times_ms), False),
                   "Band-power": (features.bandpower(ep.X), False)}
            splits = protocol.make_splits(ep.y, ep.groups, seeds)
            preds = {}
            for e, (B, pca) in enc.items():
                def fn(seed, fold, tr, te, B=B, pca=pca):
                    if pca:
                        P = features.BlockPCA().fit(B[tr])
                        E = P.transform(B)
                    else:
                        E = B
                    E = E.reshape(len(ep.y), -1)
                    return {"clean": (E[tr], E[te], None)}
                preds[e] = protocol.pooled_oof(fn, ep.y, splits)["clean"]
            ba = protocol.bootstrap_ba(ep.y, ep.groups, preds, n_boot)
            con = {f"BSC6_minus_{e}": protocol.paired_diff(ep.y, ep.groups, preds["BSC6"], preds[e], n_boot)
                   for e in enc if e != "BSC6"}
            out["results"][f"{name}|{norm}"] = {"chance": 1.0 / len(set(ep.y)), "BA": ba, "contrasts": con}
            print(f"[{name}|{norm}] " + " ".join(f"{e}={ba[e]['BA']:.3f}" for e in enc) + " | " +
                  " ".join(f"{k[11:]}:{v['mean_diff']:+.3f}[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                           for k, v in con.items()), flush=True)
    write(out, args.out or "encoder_contrasts.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
