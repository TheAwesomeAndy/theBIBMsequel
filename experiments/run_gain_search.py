#!/usr/bin/env python3
"""Nested selection of the reservoir input gain (the one free scale parameter of BSC6).

The linear encoders are standardized by the readout, so a global change of input scale
does not affect them. The LIF reservoir has a fixed threshold, so its firing depends on
input amplitude, and every earlier result used one inherited gain. Here the gain is
chosen per outer training fold from GAINS by an inner participant-grouped CV on the
training participants only; the chosen gain is refit on the whole outer training set and
scored on the untouched test participants. Reported: nested BSC6 balanced accuracy, the
distribution of chosen gains, and paired contrasts against Amp6, the LPP score, and the
fixed gain of 1.

    SEQUEL_DATA=/path python experiments/run_gain_search.py erpcore_P3 erpcore_N400 \
        erpcore_N170 tcrzem shape --shape-dir /path/batch_data
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from common import base_parser, cv_settings, protocol, write  # noqa: E402
from sequel import config as C, datasets, features, reservoir  # noqa: E402

GAINS = (0.25, 0.5, 1.0, 2.0, 4.0)


def embed_fit_predict(B, y, tr, te, seed):
    P = features.BlockPCA().fit(B[tr])
    E = P.transform(B).reshape(len(B), -1)
    sc, clf = protocol.fit_readout(E[tr], y[tr], seed)
    return clf.predict_proba(sc.transform(E[te]))


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("names", nargs="+")
    p.add_argument("--norm", default="global_scale")
    p.add_argument("--inner-folds", type=int, default=3)
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    out = {"design": vars(args) | {"gains": GAINS, "cv_seeds": list(seeds), "n_boot": n_boot}, "results": {}}
    for name in args.names:
        t0 = time.time()
        s = datasets.shape(args.shape_dir) if name == "shape" else datasets.cached(name)
        ep = s.to_epochs("fir", args.norm)
        y, g = ep.y, ep.groups
        classes = np.unique(y)
        win = C.window_samples(C.RES_WINDOWS_REL_MS["post"], onset=ep.onset)
        codes = {gain: reservoir.LIFReservoir().encode(ep.X * gain, win, 6).counts for gain in GAINS}
        amp = features.amplitude_bins(ep.X, win, 6).reshape(len(y), -1)
        lpp = features.lpp_window(ep.X, ep.times_ms).reshape(len(y), -1)
        acc = {k: np.zeros((len(y), classes.size)) for k in ("BSC6-nested", "BSC6-gain1", "Amp6", "LPP-300-1000")}
        chosen = []
        for seed, fold, tr, te in protocol.make_splits(y, g, seeds):
            inner = StratifiedGroupKFold(args.inner_folds, shuffle=True, random_state=seed)
            score = {}
            for gain in GAINS:
                pr = np.zeros((len(tr), classes.size))
                for itr, ite in inner.split(np.zeros((len(tr), 1)), y[tr], groups=g[tr]):
                    sub = codes[gain][tr]
                    pr[ite] = embed_fit_predict(sub, y[tr], itr, ite, seed)
                score[gain] = balanced_accuracy_score(y[tr], classes[pr.argmax(1)])
            best = max(GAINS, key=lambda k: (score[k], -abs(np.log(k))))   # ties -> closest to 1
            chosen.append(best)
            acc["BSC6-nested"][te] += embed_fit_predict(codes[best], y, tr, te, seed)
            acc["BSC6-gain1"][te] += embed_fit_predict(codes[1.0], y, tr, te, seed)
            for k, F in (("Amp6", amp), ("LPP-300-1000", lpp)):
                sc, clf = protocol.fit_readout(F[tr], y[tr], seed)
                acc[k][te] += clf.predict_proba(sc.transform(F[te]))
        preds = {k: classes[v.argmax(1)] for k, v in acc.items()}
        ba = protocol.bootstrap_ba(y, g, preds, n_boot)
        con = {f"nested_minus_{k}": protocol.paired_diff(y, g, preds["BSC6-nested"], preds[k], n_boot)
               for k in ("BSC6-gain1", "Amp6", "LPP-300-1000")}
        out["results"][name] = {"chance": 1.0 / classes.size, "BA": ba, "contrasts": con,
                                "chosen_gains": {str(k): v for k, v in Counter(chosen).items()},
                                "seconds": round(time.time() - t0)}
        print(f"[{name}] " + " ".join(f"{k}={v['BA']:.3f}" for k, v in ba.items()) + " | gains "
              + str(dict(Counter(chosen))) + " | " +
              " ".join(f"{k}:{v['mean_diff']:+.3f}[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for k, v in con.items()),
              flush=True)
    write(out, args.out or "gain_search.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
