"""X01b: within-subject permutation null for the BIBM reservoir under pre_sd and epoch_z.

Question: is reservoir BA 0.294 under pre-stimulus scaling a below-chance
structure, or within the null of a label-free code? The null keeps the
features, the folds of partition seed 42, and the readout, and permutes each
subject's three condition labels jointly.

Usage: SHAPE_DATA_DIR=... python experiments/X01/null_reservoir.py [n_perm]
Output: outputs/X01/x01b_reservoir_null.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

OUT = ROOT / "outputs" / "X01" / "x01b_reservoir_null.json"


def main(n_perm: int = 50):
    t0 = time.time()
    raw, y, g = cm.load_shape_raw()
    Xb = cm.downsample(raw)
    out = {}
    for norm in ["pre_sd", "epoch_z"]:
        F, _ = cm.reservoir_bsc(cm.normalize(Xb, norm))
        keep = cm.SEEDS
        cm.SEEDS = [42]
        obs = cm.balanced_accuracy(y, cm.oof_proba(F, y, g, block_pca_k=64).argmax(1))
        cm.SEEDS = keep
        d = cm.label_permutation_null(F, y, g, n_perm=n_perm, block_pca_k=64)
        out[norm] = {"observed_seed42": obs, "null_mean": float(d.mean()), "null_sd": float(d.std()),
                     "null_q025": float(np.percentile(d, 2.5)), "null_q975": float(np.percentile(d, 97.5)),
                     "p_lower": float((1 + np.sum(d <= obs)) / (1 + len(d))),
                     "p_upper": float((1 + np.sum(d >= obs)) / (1 + len(d))), "null": d.tolist()}
        print(norm, {k: v for k, v in out[norm].items() if k != "null"}, f"{time.time() - t0:.0f}s", flush=True)
    cm.write_json(OUT, {"provenance": cm.provenance("X01b-2026-11", {"n_perm": n_perm, "partition_seed": 42}),
                        **out})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
