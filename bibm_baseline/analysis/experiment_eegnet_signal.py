#!/usr/bin/env python3
"""Camera-ready experiment E3: EEGNet under the signal-level channel-removal protocol.

EEGNet-8,2 (Lawhern et al., 2018) is trained per fold on the raw 34x256 ERP signal,
with and without train-time augmentation (random channel dropout and additive
amplitude noise), and evaluated on the signal-level conditions of perturbations.py: electrodes removed
from the raw test signal at 0/10/30/50%, 5 dB additive noise, and +/-50 ms jitter. The removed channels in each (seed, fold) are drawn exactly as in
analysis/experiment2_rawsignal.py (np.random.default_rng(1000*seed + fold)), so EEGNet
and the fixed encoders lose the same electrodes. A removed electrode is a zero raw
trace; train-only per-channel standardization is then applied as for clean data.

Inference matches the fixed encoders: out-of-fold class probabilities are averaged
per observation over the five repeated partitions (seeds 42-46), then 95% CIs come
from a subject-level bootstrap (211 subjects, three observations each, n_boot 4000).
For a matched comparison the ERP-window encoder is recomputed under the same
removals, and paired subject-bootstrap differences EEGNet - ERP-window are reported.

Reads the restricted SHAPE pickle locally; writes only aggregate JSON.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis_subject_bootstrap as RB  # noqa: E402
import experiment2_rawsignal as E2S  # noqa: E402
import perturbations as PT  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
LEVELS = [0.0, 0.1, 0.3, 0.5]
EPOCHS, BATCH, LR, DROPOUT = 80, 64, 1e-3, 0.25
torch.set_num_threads(2)


class EEGNet(nn.Module):
    """EEGNet-8,2: F1=8, D=2, F2=16, kernel 64, for 34 channels x 256 samples."""

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


def augment(xb, gen):
    """Per sample: with prob 0.5 zero 0-50% of channels; add Gaussian noise of std U(0, 0.5)."""
    n, _, c, _ = xb.shape
    k = torch.randint(0, c // 2 + 1, (n, 1), generator=gen)
    k = k * (torch.rand(n, 1, generator=gen) < 0.5)                     # half the samples untouched
    rank = torch.argsort(torch.rand(n, c, generator=gen), dim=1).argsort(dim=1)
    keep = (rank >= k).float()[:, None, :, None]                         # drop k random channels
    sd = torch.rand(n, 1, 1, 1, generator=gen) * 0.5
    return xb * keep + sd * torch.randn(xb.shape, generator=gen)


def train_eegnet(Xtr, ytr, seed, aug):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = EEGNet(); opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    X = torch.tensor(Xtr, dtype=torch.float32); Y = torch.tensor(ytr, dtype=torch.long)
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]; xb = X[idx]
            if aug:
                xb = augment(xb, gen)
            opt.zero_grad(); lossf(model(xb), Y[idx]).backward(); opt.step()
    model.eval()
    return model


def to_input(raw, mu, sd):
    """raw (n,256,34) -> standardized (n,1,34,256)."""
    return ((raw - mu) / sd).transpose(0, 2, 1)[:, None, :, :]


def conditions(X, seed, fold):
    """Yield (name, corrupted signal) for every signal-level condition, shared draws (perturbations.py)."""
    for frac in LEVELS:
        Xc, _ = PT.remove_channels(X, frac, seed, fold)
        yield f"remove_{frac:.1f}", Xc
    for name, fn in PT.SIGNAL_CONDITIONS.items():
        yield name, fn(X, seed, fold)


def main():
    d = pickle.load(open(RB._resolve_pickle(), "rb"))
    X = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    N, T, C = X.shape; classes = np.unique(y)
    conds = [f"remove_{f:.1f}" for f in LEVELS] + list(PT.SIGNAL_CONDITIONS)
    models_ = ("EEGNet", "EEGNet+aug", "ERP-window")
    acc = {(m, c): np.zeros((N, classes.size)) for m in models_ for c in conds}
    t0 = time.time()
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((N, 1)), y, groups=g)):
            mu = X[tr].mean(axis=(0, 1)); sd = X[tr].std(axis=(0, 1)) + 1e-8
            nets = {"EEGNet": train_eegnet(to_input(X[tr], mu, sd), y[tr], 100 * seed + fold, False),
                    "EEGNet+aug": train_eegnet(to_input(X[tr], mu, sd), y[tr], 100 * seed + fold, True)}
            erp_tr = E2S.erp_all(X[tr]).reshape(len(tr), -1)
            for cname, Xte in conditions(X[te], seed, fold):
                xin = torch.tensor(to_input(Xte, mu, sd), dtype=torch.float32)
                for name, net in nets.items():
                    with torch.no_grad():
                        acc[(name, cname)][te] += torch.softmax(net(xin), 1).numpy()
                E2S._fit_pred(erp_tr, y[tr], E2S.erp_all(Xte).reshape(len(te), -1),
                              seed, acc[("ERP-window", cname)], te, classes)
            print(f"[E3] seed {seed} fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
    preds = {k: classes[np.argmax(v, axis=1)] for k, v in acc.items()}
    res = {"protocol": "EEGNet-8,2 (F1=8,D=2,F2=16,k=64,dropout=%.2f), Adam lr=%g, %d epochs, batch %d; "
           "aug: per-sample p=0.5 zero 0-50%% channels + Gaussian noise std~U(0,0.5); "
           "signal-level conditions from perturbations.py (removal draw as in experiment2_rawsignal.py; "
           "5 dB additive noise; +/-50 ms alignment jitter); StratifiedGroupKFold(5) x seeds 42-46; "
           "train-only per-channel z-score; OOF proba averaged over partitions; subject-level "
           "bootstrap n_boot=%d" % (DROPOUT, LR, EPOCHS, BATCH, RB.N_BOOT),
           "BA": {}, "paired_minus_ERPwindow": {}}
    for c in conds:
        pm = {m: preds[(m, c)] for m in models_}
        pt, ci, _ = RB.subject_bootstrap(y, g, pm)
        res["BA"][c] = {m: {"BA": pt[m], "ci95": list(ci[m])} for m in pm}
        res["paired_minus_ERPwindow"][c] = {}
        for m in ("EEGNet", "EEGNet+aug"):
            md, lo, hi, _ = RB.paired_diff_ci(y, g, pm[m], pm["ERP-window"])
            res["paired_minus_ERPwindow"][c][m] = {"mean_diff": md, "ci95": [lo, hi]}
        print(f"[E3 {c}] " + " | ".join(f"{m}={pt[m]:.3f}[{ci[m][0]:.3f},{ci[m][1]:.3f}]" for m in pm), flush=True)
    # training effect and losses as paired subject-level contrasts under the same resampling
    res["paired_aug_minus_unaug"] = {}
    res["paired_clean_minus_condition"] = {}
    for c in conds:
        md, lo, hi, _ = RB.paired_diff_ci(y, g, preds[("EEGNet+aug", c)], preds[("EEGNet", c)])
        res["paired_aug_minus_unaug"][c] = {"mean_diff": md, "ci95": [lo, hi]}
        if c == "remove_0.0":
            continue
        res["paired_clean_minus_condition"][c] = {}
        for m in ("EEGNet", "EEGNet+aug"):
            md, lo, hi, _ = RB.paired_diff_ci(y, g, preds[(m, "remove_0.0")], preds[(m, c)])
            res["paired_clean_minus_condition"][c][m] = {"mean_diff": md, "ci95": [lo, hi]}
        print(f"[E3 paired {c}] aug-unaug={res['paired_aug_minus_unaug'][c]['mean_diff']:+.3f} "
              f"[{res['paired_aug_minus_unaug'][c]['ci95'][0]:+.3f},{res['paired_aug_minus_unaug'][c]['ci95'][1]:+.3f}]",
              flush=True)
    out = Path(os.environ.get("E3_OUT", Path(__file__).resolve().parents[1] / "outputs" / "aggregate"
                              / "e3_eegnet_signal.json"))
    json.dump(res, open(out, "w"), indent=2)
    print(f"[out] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
