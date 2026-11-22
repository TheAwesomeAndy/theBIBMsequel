"""X11b-2026-11 (Paper B): the X08 panel controls on the 211-participant cohort of X11.

X08 used the X07 loader without exclusion (212 participants). One of them has an
all-zero epoch and is excluded in X11. This run repeats the X08 test-side rules,
unscaled input, with that participant removed, so that every SHAPE number in
Paper B refers to the same 211 participants. The X08 output is left unchanged.

Usage: SHAPE_DATA_DIR=... python experiments/X11/panels.py
Output: outputs/X11/x11_panels.json
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "X07"))
from sequel import common as cm  # noqa: E402
import run as x07  # noqa: E402

spec = importlib.util.spec_from_file_location("x08", ROOT / "experiments" / "X08" / "run.py")
x08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x08)

t0 = time.time()
X, y, g = x07.load_decimated(); X = x07.scale(X, "none")
tr = np.einsum("ntc,ntc->n", X, X)
keep = ~np.isin(g, np.unique(g[tr <= 0])); X, y, g = X[keep], y[keep], g[keep]
F = X.reshape(len(X), -1)
res = {rule: x08.run(F, y, g, rule) for rule in ["uncentered", "full_panel", "loo"]}
for rule in ["panel2", "panel1"]:
    vals = [x08.run(F, y, g, rule, rng=np.random.default_rng(d)) for d in range(5)]
    res[rule] = float(np.mean(vals)); res[rule + "_draws"] = vals
rng = np.random.default_rng(2026); null = []
for _ in range(100):
    yp = y.copy()
    for s in np.unique(g):
        i = np.flatnonzero(g == s); yp[i] = rng.permutation(y[i])
    null.append(x08.run(F, y, g, "full_panel", yperm=yp))
res["null_full_panel"] = {"n_perm": 100, "mean": float(np.mean(null)), "max": float(np.max(null)),
                          "p": float((1 + np.sum(np.array(null) >= res["full_panel"])) / 101)}
cm.write_json(ROOT / "outputs" / "X11" / "x11_panels.json", {
    "provenance": cm.provenance("X11b-2026-11", {"rules": "experiments/X08/run.py", "input": "unscaled, X07 loader, one participant with an all-zero epoch excluded"}),
    "n_subjects": int(len(np.unique(g))), "results": res, "runtime_s": round(time.time() - t0, 1)})
print(len(np.unique(g)), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not k.endswith("draws")})
