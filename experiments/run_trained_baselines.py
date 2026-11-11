#!/usr/bin/env python3
"""Phase 3. Trained baselines under the same protocol: EEGNet (with and without
augmentation) and Riemannian tangent space (ERP covariances, xDAWN covariances).

Per dataset, every model sees the post-stimulus window (0-800 ms) of the same analysed
signal as the fixed encoders, the same participant-grouped splits, and the same removed
electrodes (draws keyed to seed and fold). Conditions:
    clean           intact test signal
    sig{p}_zero     p% of electrodes zeroed in the test signal
    sig{p}_spline   the same electrodes repaired by spherical-spline interpolation (needs positions)
    refit{p}        model retrained on the retained electrodes (Riemannian only; EEGNet is
                    too costly to retrain per condition)
Amp6 and the LPP score are computed in the same loop so that paired contrasts share
resamples. Output: one aggregate JSON.

    SEQUEL_DATA=/path python experiments/run_trained_baselines.py erpcore_P3 erpcore_N400 \
        erpcore_N170 tcrzem --jobs 4
    python experiments/run_trained_baselines.py --synthetic --quick --epochs 3   # code check
"""
from __future__ import annotations

import sys
import time

import numpy as np
from joblib import Parallel, delayed

from common import base_parser, cv_settings, protocol, removal_draw, write  # noqa: E402
from sequel import config as C, datasets, features, fills, trained  # noqa: E402

ALL_MODELS = ("EEGNet", "EEGNet+aug", "TS-ERPCov", "TS-xDAWN", "Amp6", "LPP-300-1000")


def _linear(Ftr, ytr, Fte_list, seed):
    sc, clf = protocol.fit_readout(Ftr.reshape(len(Ftr), -1), ytr, seed)
    return [clf.predict_proba(sc.transform(F.reshape(len(F), -1))) for F in Fte_list]


def run_fold(seed, fold, tr, te, Xw, Xfull, times, win, y, positions, rate, models, epochs):
    """Return {(model, condition): (te, proba)} for one split."""
    pct = int(round(rate * 100))
    n_ch = Xw.shape[2]
    drop = removal_draw(n_ch, rate, seed, fold)
    keep = np.setdiff1d(np.arange(n_ch), drop)
    tests = {"clean": (Xw[te], Xfull[te])}
    Xz, Xfz = Xw[te].copy(), Xfull[te].copy()
    Xz[:, :, drop] = 0.0
    Xfz[:, :, drop] = 0.0
    tests[f"sig{pct}_zero"] = (Xz, Xfz)
    if positions is not None:
        tests[f"sig{pct}_spline"] = (fills.interpolate_signal(Xw[te], drop, positions),
                                     fills.interpolate_signal(Xfull[te], drop, positions))
    out = {}
    s = 100 * seed + fold
    for m in models:
        if m in ("EEGNet", "EEGNet+aug"):
            clf = trained.EEGNetClassifier(s, aug=(m == "EEGNet+aug"), epochs=epochs).fit(Xw[tr], y[tr])
            for c, (xw, _) in tests.items():
                out[(m, c)] = (te, clf.predict_proba(xw))
        elif m.startswith("TS-"):
            kind = "erpcov" if m == "TS-ERPCov" else "xdawn"
            clf = trained.RiemannClassifier(kind, s).fit(Xw[tr], y[tr])
            for c, (xw, _) in tests.items():
                out[(m, c)] = (te, clf.predict_proba(xw))
            rclf = trained.RiemannClassifier(kind, s).fit(Xw[tr][:, :, keep], y[tr])
            out[(m, f"refit{pct}")] = (te, rclf.predict_proba(Xw[te][:, :, keep]))
        else:
            enc = (lambda X: features.amplitude_bins(X, win, 6)) if m == "Amp6" else \
                  (lambda X: features.lpp_window(X, times))
            names = list(tests)
            probs = _linear(enc(Xfull[tr]), y[tr], [enc(tests[c][1]) for c in names], seed)
            for c, p in zip(names, probs):
                out[(m, c)] = (te, p)
            pr = _linear(enc(Xfull[tr][:, :, keep]), y[tr], [enc(Xfull[te][:, :, keep])], seed)[0]
            out[(m, f"refit{pct}")] = (te, pr)
    return out


def run_dataset(s, args, seeds, n_boot):
    ep = s.to_epochs("fir", args.norm)
    y, g = ep.y, ep.groups
    win = C.window_samples(C.RES_WINDOWS_REL_MS["post"], onset=ep.onset)
    Xw = ep.X[:, win[0]:win[1], :]
    splits = protocol.make_splits(y, g, seeds)
    t0 = time.time()
    parts = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(sd, fd, tr, te, Xw, ep.X, ep.times_ms, win, y, s.positions, args.rate,
                          args.models, args.epochs) for sd, fd, tr, te in splits)
    classes = np.unique(y)
    acc = {}
    for part in parts:
        for key, (te, p) in part.items():
            acc.setdefault(key, np.zeros((len(y), classes.size)))[te] += p
    preds = {k: classes[np.argmax(v, axis=1)] for k, v in acc.items()}
    pct = int(round(args.rate * 100))
    conds = sorted({c for _, c in preds})
    res = {"chance": 1.0 / classes.size, "seconds": round(time.time() - t0),
           "BA": {m: protocol.bootstrap_ba(y, g, {c: preds[(m, c)] for c in conds if (m, c) in preds}, n_boot)
                  for m in args.models},
           "paired": {}}
    ref = [r for r in ("Amp6", "LPP-300-1000") if r in args.models]
    for m in args.models:
        for r in ref:
            if r == m:
                continue
            for c in ("clean", f"sig{pct}_zero"):
                res["paired"][f"{m}_minus_{r}|{c}"] = protocol.paired_diff(y, g, preds[(m, c)], preds[(r, c)], n_boot)
        if (m, f"sig{pct}_zero") in preds:
            res["paired"][f"{m}|clean_minus_sig{pct}_zero"] = protocol.paired_diff(
                y, g, preds[(m, "clean")], preds[(m, f"sig{pct}_zero")], n_boot)
    if "EEGNet" in args.models and "EEGNet+aug" in args.models:
        for c in conds:
            if ("EEGNet", c) in preds:
                res["paired"][f"aug_minus_unaug|{c}"] = protocol.paired_diff(
                    y, g, preds[("EEGNet+aug", c)], preds[("EEGNet", c)], n_boot)
    return res


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("names", nargs="*", default=[])
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    p.add_argument("--norm", default="global_scale")
    p.add_argument("--rate", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=trained.EPOCHS)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    seeds, n_boot = cv_settings(args)
    out = {"design": vars(args) | {"cv_seeds": list(seeds), "n_boot": n_boot,
                                   "eegnet_recipe": "BIBM camera-ready (F1=8, D=2, F2=16, k=64, dropout 0.25, "
                                                    "Adam 1e-3, batch 64)"}, "datasets": {}}
    if args.synthetic:
        from tests.synthetic import electrode_positions, raw_epochs
        raw, y, g = raw_epochs(n_subj=24)
        sets = {"synthetic": datasets.ERPSet("synthetic", raw, y, g, 1024, 205, [f"c{i}" for i in range(34)],
                                             ["a", "b", "c"], electrode_positions(34))}
    else:
        sets = {n: (datasets.shape(args.shape_dir) if n == "shape" else datasets.cached(n)) for n in args.names}
    for name, s in sets.items():
        print(f"[{name}] {s.summary()}", flush=True)
        r = run_dataset(s, args, seeds, n_boot)
        out["datasets"][name] = {"summary": s.summary(), "results": r}
        for m in args.models:
            b = r["BA"][m]
            print(f"  [{m}] " + " ".join(f"{c}={v['BA']:.3f}" for c, v in b.items()), flush=True)
    write(out, args.out or "p3_trained_baselines.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
