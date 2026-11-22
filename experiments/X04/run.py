"""X04-2026-11: does a trained EEGNet learn the normalization leak? (SHAPE)

EEGNet-8,2 with the accepted BIBM recipe (analysis/experiment_eegnet_signal.py:
F1=8, D=2, F2=16, dropout 0.25, Adam lr 1e-3, 80 epochs, batch 64, torch seed
100*seed+fold, train-only per-channel standardization, out-of-fold
probabilities summed over the five seeded partitions).

Inputs, each under epoch_z and pre_sd:
  full : the analysed epoch, 256 samples (-200..+800 ms)
  pre  : samples [0, 51), i.e. -200..0 ms only (temporal kernel 32)

Prediction, fixed before this run: EEGNet on `pre` is above chance under
epoch_z and at chance under pre_sd. Anchor: BIBM reports EEGNet BA 0.731 on
`full` under epoch_z.

Run record: the first run finished pre|epoch_z (0.539), full|epoch_z (0.731)
and pre|pre_sd (0.300) and was stopped by a session interruption during
full|pre_sd. Its predictions were not saved, so the resumed run recomputes
all four; training is seeded, so the three finished values must reproduce.

Usage: SHAPE_DATA_DIR=... python experiments/X04/run.py [RUN ...]
Output: outputs/X04/x04_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sequel import common as cm  # noqa: E402

EPOCHS, BATCH, LR, DROPOUT = 80, 64, 1e-3, 0.25
torch.set_num_threads(1)
OUT = ROOT / "outputs" / "X04" / "x04_results.json"
PRED = Path("/home/claude/data_local/x04_oof_predictions.json")  # out-of-fold labels, kept outside the repo


class EEGNet(nn.Module):
    def __init__(self, n_ch=34, n_t=256, n_cls=3, F1=8, D=2, F2=16, k=64, p=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p), nn.Flatten())
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_flat, n_cls)

    def forward(self, x):
        return self.fc(self.net(x))


def train(Xtr, ytr, seed, n_t, k):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = EEGNet(n_t=n_t, k=k); opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    X = torch.tensor(Xtr, dtype=torch.float32); Y = torch.tensor(ytr, dtype=torch.long)
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(X[idx]), Y[idx]).backward(); opt.step()
    return model.eval()


def run(X, y, g, k):
    N, T, C = X.shape
    P = np.zeros((N, 3))
    for seed in cm.SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(N), y, g)):
            mu = X[tr].mean(axis=(0, 1)); sd = X[tr].std(axis=(0, 1)) + 1e-8
            f = lambda A: ((A - mu) / sd).transpose(0, 2, 1)[:, None]
            net = train(f(X[tr]), y[tr], 100 * seed + fold, T, k)
            with torch.no_grad():
                P[te] += torch.softmax(net(torch.tensor(f(X[te]), dtype=torch.float32)), 1).numpy()
    return P.argmax(1)


def main(only=None):
    """`only`: list like ['full|pre_sd']; predictions of other runs are reloaded from PRED."""
    t0 = time.time()
    raw, y, g = cm.load_shape_raw()
    Xb = cm.downsample(raw)
    preds = {k: np.array(v) for k, v in json.loads(PRED.read_text()).items()} if (only and PRED.exists()) else {}
    for norm in ["epoch_z", "pre_sd"]:
        Xn = cm.normalize(Xb, norm)
        for name, Xi, k in [("pre", Xn[:, :cm.ONSET_DS], 32), ("full", Xn, 64)]:
            if only and f"{name}|{norm}" not in only:
                continue
            preds[f"eegnet_{name}|{norm}"] = run(Xi, y, g, k)
            PRED.write_text(json.dumps({kk: vv.tolist() for kk, vv in preds.items()}))
            print(f"{name}|{norm} BA {cm.balanced_accuracy(y, preds[f'eegnet_{name}|{norm}']):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    summary, B, names = cm.subject_bootstrap(y, g, preds)
    contrasts = {f"{n}: epoch_z - pre_sd": cm.paired(B, names, f"eegnet_{n}|epoch_z", f"eegnet_{n}|pre_sd")
                 for n in ("pre", "full")}
    cm.write_json(OUT, {"provenance": cm.provenance("X04-2026-11", {
        "model": "EEGNet-8,2", "epochs": EPOCHS, "batch": BATCH, "lr": LR, "dropout": DROPOUT,
        "pre_kernel": 32, "full_kernel": 64, "torch": torch.__version__}),
        "balanced_accuracy": summary, "contrasts": contrasts, "runtime_s": round(time.time() - t0, 1)})
    print(summary, contrasts)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
