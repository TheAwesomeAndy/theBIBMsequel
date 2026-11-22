"""Summary of X12 against the reference-size law and the break-even corollary.

Reads outputs/X12/x12_results.json and writes outputs/X12/x12_law_summary.json.
No model is fitted here: the predicted curve uses the center_all accuracy only
(Proposition 1), and the predicted break-even m* uses the uncentered and
center_all accuracies only (Corollary 1), both in probit units.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "outputs/X12/x12_results.json").read_text())
ms = d["provenance"]["ms"]; out = {}; err = []
for par, r in d["results"].items():
    obs = np.array([r[f"center_ref_{m}"]["ba"] for m in ms])
    pred = np.array([r["law"][str(m)]["predicted_ba"] for m in ms])
    err += list(obs - pred)
    z0, zinf = norm.ppf(r["uncentered"]["ba"]), norm.ppf(r["center_all"]["ba"])
    ratio = (z0 / zinf) ** 2
    m_star = float(ratio / (1 - ratio)) if ratio < 1 else float("inf")
    above = [m for m, o in zip(ms, obs) if o > r["uncentered"]["ba"]]
    out[par] = {"uncentered": r["uncentered"]["ba"], "center_all": r["center_all"]["ba"],
                "gain_center_all": r["gain_center_all"], "gain_ea": r["gain_ea"],
                "r_probit": float(ratio), "m_star_predicted": m_star,
                "smallest_tested_m_above_uncentered": (min(above) if above else None),
                "law_abs_error_mean": float(np.abs(obs - pred).mean()), "law_abs_error_max": float(np.abs(obs - pred).max()),
                "law_error_at_m1": float(obs[0] - pred[0])}
err = np.array(err)
summary = {"n_points": int(len(err)), "abs_error_mean": float(np.abs(err).mean()), "abs_error_max": float(np.abs(err).max()),
           "abs_error_mean_m_ge_2": float(np.mean([abs(e) for i, e in enumerate(err) if i % len(ms) != 0])),
           "signed_error_mean": float(err.mean()), "per_paradigm": out}
(ROOT / "outputs/X12/x12_law_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
