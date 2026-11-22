"""X10-2026-11: permutation resolution and multiplicity for the ERP CORE across-participant test.

X02 used 200 permutations, so its smallest attainable p is 0.005, and it reported
five paradigms without a multiplicity correction. This run recomputes the same
pre-registered statistic (X02, across regime, trials scaled then averaged) with
2000 permutations per paradigm for `epoch_z` and 500 for `none`, and reports Holm-adjusted
p values over the five paradigms. The observed statistic must equal the X02 value.
The X02 output is left unchanged.

Usage: ERPCORE_DATA_DIR=... python experiments/X10/run.py
Output: outputs/X10/x10_results.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402
from sequel import erpcore as ec  # noqa: E402

spec = importlib.util.spec_from_file_location("x02", ROOT / "experiments" / "X02" / "run.py")
x02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x02)

N_PERM = {"epoch_z": 2000, "none": 500}
NORMS = ["epoch_z", "none"]
OUT = ROOT / "outputs" / "X10" / "x10_results.json"


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    t0 = time.time()
    ref = json.loads((ROOT / "outputs" / "X02" / "x02_results.json").read_text())["results"]
    res = json.loads(OUT.read_text())["results"] if OUT.exists() else {}
    for pi, par in enumerate(ec.CONTRASTS):
        if par in res:
            continue
        rng = np.random.default_rng([1010, pi])
        A = {n: [] for n in NORMS}; g, y2 = [], []; centers = None
        for pid, f in ec.participants(par):
            x, times, y, _ = ec.load(f, par)
            if min((y == 0).sum(), (y == 1).sum()) == 0:
                continue
            centers = times[:(len(times) // x02.BIN) * x02.BIN].reshape(-1, x02.BIN).mean(1)
            for n in NORMS:
                xb = x02.binned(ec.normalize(x, times, n))
                A[n] += [xb[y == c].mean(0) for c in (0, 1)]
            g += [pid, pid]; y2 += [0, 1]
        g = np.array(g); y2 = np.array(y2)
        pre = np.flatnonzero((centers >= -200) & (centers < 0))
        out = {}
        for n in NORMS:
            Arr = np.stack(A[n])
            obs = float(x02.across(Arr, y2, g, pre).mean())
            null = np.empty(N_PERM[n])
            for k in range(N_PERM[n]):
                flip = rng.random(len(g) // 2) < 0.5
                yp = y2.reshape(-1, 2).copy(); yp[flip] = yp[flip][:, ::-1]
                null[k] = x02.across(Arr, yp.ravel(), g, pre, repeats=1, seed=int(rng.integers(1e6))).mean()
            out[n] = {"pre_mean": obs, "x02_value": ref[par]["across"][n]["pre_mean"],
                      "p": float((1 + np.sum(null >= obs)) / (1 + N_PERM[n])), "n_perm": N_PERM[n],
                      "null_mean": float(null.mean()), "null_sd": float(null.std()),
                      "null_q95": float(np.percentile(null, 95)), "null_max": float(null.max()),
                      "z_vs_null": float((obs - null.mean()) / null.std())}
            print(f"{par:5s} {n:8s} {obs:.3f} (X02 {out[n]['x02_value']:.3f}) p={out[n]['p']:.4f} "
                  f"null max {null.max():.3f} {time.time() - t0:.0f}s", flush=True)
        res[par] = out
        pars = list(res)
        adj = {n: dict(zip(pars, holm([res[p][n]["p"] for p in pars]))) for n in NORMS} if len(pars) == len(ec.CONTRASTS) else {}
        cm.write_json(OUT, {"provenance": {"experiment": "X10-2026-11", "n_perm": N_PERM, "norms": NORMS,
                                           "statistic": "X02 across regime, mean BA over pre-stimulus bins",
                                           "null": "within-participant label swaps, one CV repeat per permutation (observed uses five, which makes the test conservative)"},
                            "results": res, "holm_adjusted_p": adj, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
