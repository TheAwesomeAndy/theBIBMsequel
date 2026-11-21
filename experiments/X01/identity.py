"""X01c: Proposition 1 on SHAPE.

Compares the pre-stimulus mean of each z-scored epoch and channel with -m/sd of
the unscaled epoch: correlation, and decoding agreement under the BIBM
protocol.

Usage: SHAPE_DATA_DIR=... python experiments/X01/identity.py
Output: outputs/X01/x01c_identity.json
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

raw, y, g = cm.load_shape_raw()
X = cm.downsample(raw)
Z = cm.normalize(X, "epoch_z")
A = Z[:, 9:cm.ONSET_DS, :].mean(1)
B = -X.mean(1) / X.std(1)
pa = cm.oof_proba(A, y, g).argmax(1)
pb = cm.oof_proba(B, y, g).argmax(1)
out = {
    "provenance": cm.provenance("X01c-2026-11", {"pre_window_ds": [9, cm.ONSET_DS]}),
    "pearson_r": float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
    "ba_prestim_mean_z": cm.balanced_accuracy(y, pa),
    "ba_neg_m_over_sd": cm.balanced_accuracy(y, pb),
    "prediction_agreement": float((pa == pb).mean()),
    "max_abs_prestim_mean_unscaled_uV": float(np.abs(X[:, :cm.ONSET_DS, :].mean(1)).max()),
}
cm.write_json(ROOT / "outputs" / "X01" / "x01c_identity.json", out)
print({k: v for k, v in out.items() if k != "provenance"})
