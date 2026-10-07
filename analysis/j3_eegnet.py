#!/usr/bin/env python3
"""J3: EEGNet-8,2 with and without augmentation on the external TCRZEM cohort.

The conference recipe (experiment_eegnet_signal.py: EEGNet-8,2, Adam lr 1e-3, 80 epochs,
batch 64, dropout 0.25; augmentation = per sample with p 0.5 zero 0-50% of channels plus
Gaussian noise with SD ~ U(0, 0.5)) trained per fold on the raw 32 x 256 signal with
train-only per-channel standardization, evaluated on the signal-level conditions of J2
(same draws): clean; removal 10/30/50%; spline repair 30/50%; 5 dB noise; +/-50 ms
jitter; re-averaging from 1/4/16 trials. The ERP-window encoder is recomputed under the
same conditions for paired contrasts.

Optional recipe sensitivity (environment variable J3_RECIPE): "default", "epochs40",
"epochs160", "dropout50". Each recipe writes its own JSON. Recipe variants run on the
seed-42 partition only (J3_SEEDS, default "42" for variants and "42,43,44,45,46" for the
default recipe); the JSON records the seeds used. Writes aggregate JSON only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcore as J  # noqa: E402
import perturbations as PT  # noqa: E402
import spline as SP  # noqa: E402
import tcrzem_data as TD  # noqa: E402
from j2_signal_level import trial_average  # noqa: E402

RECIPES = {"default": (80, 0.25), "epochs40": (40, 0.25), "epochs160": (160, 0.25),
           "dropout50": (80, 0.50)}
RECIPE = os.environ.get("J3_RECIPE", "default")
EPOCHS, DROPOUT = RECIPES[RECIPE]
BATCH, LR = 64, 1e-3
torch.set_num_threads(int(os.environ.get("J3_THREADS", "2")))
SEEDS = [int(x) for x in os.environ.get("J3_SEEDS", "42,43,44,45,46" if RECIPE == "default" else "42").split(",")]
if RECIPE == "default":
    CONDS = (["clean", "remove_0.1", "remove_0.3", "remove_0.5", "spline_0.3", "spline_0.5",
              "amp_5dB", "jitter_50ms", "trials_1", "trials_4", "trials_16"])
else:
    CONDS = ["clean", "remove_0.3", "remove_0.5", "amp_5dB"]


class EEGNet(nn.Module):
    """EEGNet-8,2: F1=8, D=2, F2=16, kernel 64 (conference definition)."""

    def __init__(self, n_ch, n_t=256, n_cls=3, F1=8, D=2, F2=16, k=64, p=DROPOUT):
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
    n, _, c, _ = xb.shape
    k = torch.randint(0, c // 2 + 1, (n, 1), generator=gen)
    k = k * (torch.rand(n, 1, generator=gen) < 0.5)
    rank = torch.argsort(torch.rand(n, c, generator=gen), dim=1).argsort(dim=1)
    keep = (rank >= k).float()[:, None, :, None]
    sd = torch.rand(n, 1, 1, 1, generator=gen) * 0.5
    return xb * keep + sd * torch.randn(xb.shape, generator=gen)


def train_eegnet(Xtr, ytr, seed, aug, n_ch):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = EEGNet(n_ch, n_t=Xtr.shape[-1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); Y = torch.tensor(ytr, dtype=torch.long)
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]; xb = Xt[idx]
            if aug:
                xb = augment(xb, gen)
            opt.zero_grad(); lossf(model(xb), Y[idx]).backward(); opt.step()
    model.eval()
    return model


def to_input(raw, mu, sd):
    return ((raw - mu) / sd).transpose(0, 2, 1)[:, None, :, :]


def main():
    t0 = time.time()
    Tr, t_subj, t_val, _ = TD.load_trials()
    Xuv, y, g = TD.average(Tr, t_subj, t_val, np.unique(t_subj), zscore=False)
    X = TD.zscore_epochs(Xuv)
    N, T, C = X.shape
    classes = np.unique(y)
    order = {}
    for i, (s, v) in enumerate(zip(t_subj, t_val)):
        order.setdefault((s, int(v)), []).append(i)
    obs_trials = [np.array(order[(s, int(v))]) for s, v in zip(g, y)]
    pos = SP.sph_to_cart(TD.montage()[1])
    models_ = ("EEGNet", "EEGNet+aug", "ERP-window")
    acc = {(m, c): np.zeros((N, classes.size)) for m in models_ for c in CONDS}
    for seed, fold, tr, te in J.folds(y, g, seeds=SEEDS):
        mu = X[tr].mean(axis=(0, 1)); sd = X[tr].std(axis=(0, 1)) + 1e-8
        nets = {"EEGNet": train_eegnet(to_input(X[tr], mu, sd), y[tr], 100 * seed + fold, False, C),
                "EEGNet+aug": train_eegnet(to_input(X[tr], mu, sd), y[tr], 100 * seed + fold, True, C)}
        erp_tr = J.erp_windows(X[tr]).reshape(len(tr), -1)
        _, sc, clf = J.fit_pred(erp_tr, y[tr], erp_tr[:1], seed, return_model=True)
        for cname in CONDS:
            if cname == "clean":
                Xte = X[te]
            elif cname.startswith("remove_"):
                Xte, _ = PT.remove_channels(X[te], float(cname.split("_")[1]), seed, fold)
            elif cname.startswith("spline_"):
                drop = PT.removal_draw(C, float(cname.split("_")[1]), seed, fold)
                Xte = TD.zscore_epochs(SP.repair(Xuv[te], pos, drop))
            elif cname.startswith("trials_"):
                n = int(cname.split("_")[1])
                rng = np.random.default_rng(30_000 + 1000 * seed + fold + 7 * n)
                Xte = TD.zscore_epochs(trial_average(Tr, obs_trials, te, n, rng))
            else:
                Xte = PT.SIGNAL_CONDITIONS[cname](X[te], seed, fold)
            xin = torch.tensor(to_input(Xte, mu, sd), dtype=torch.float32)
            for name, net in nets.items():
                with torch.no_grad():
                    acc[(name, cname)][te] += torch.softmax(net(xin), 1).numpy()
            acc[("ERP-window", cname)][te] += clf.predict_proba(
                sc.transform(J.erp_windows(Xte).reshape(len(te), -1)))
        print(f"[J3 {RECIPE}] seed {seed} fold {fold} ({time.time() - t0:.0f}s)", flush=True)
    preds = {k: J.argmax_pred(v, classes) for k, v in acc.items()}
    res = {"recipe": RECIPE, "cv_seeds": SEEDS,
           "protocol": "EEGNet-8,2 (F1=8,D=2,F2=16,k=64,dropout=%.2f), Adam lr=%g, %d epochs, batch %d; "
                       "aug: per-sample p=0.5 zero 0-50%% channels + Gaussian noise std~U(0,0.5); "
                       "J2 signal-level conditions and draws; StratifiedGroupKFold(5) x the listed seeds; "
                       "train-only per-channel z-score; OOF proba averaged over partitions; "
                       "subject-level bootstrap n_boot=%d" % (DROPOUT, LR, EPOCHS, BATCH, J.N_BOOT),
           "BA": {}, "clean_metrics": {}, "paired_minus_ERPwindow": {}, "paired_aug_minus_unaug": {},
           "paired_clean_minus_condition": {}}
    for m in ("EEGNet", "EEGNet+aug"):
        res["clean_metrics"][m] = J.clean_metrics(y, g, acc[(m, "clean")])
    for c in CONDS:
        pm = {m: preds[(m, c)] for m in models_}
        pt, ci, _ = J.boot_ci(y, g, pm)
        res["BA"][c] = {m: {"BA": pt[m], "ci95": ci[m]} for m in models_}
        res["paired_minus_ERPwindow"][c] = {m: J.paired_ci(y, g, pm[m], pm["ERP-window"])
                                            for m in ("EEGNet", "EEGNet+aug")}
        res["paired_aug_minus_unaug"][c] = J.paired_ci(y, g, pm["EEGNet+aug"], pm["EEGNet"])
        if c != "clean":
            res["paired_clean_minus_condition"][c] = {m: J.paired_ci(y, g, preds[(m, "clean")], pm[m])
                                                      for m in ("EEGNet", "EEGNet+aug")}
        print(f"[J3 {c}] " + " | ".join(f"{m}={pt[m]:.3f}[{ci[m][0]:.3f},{ci[m][1]:.3f}]" for m in models_)
              + f" aug-unaug={res['paired_aug_minus_unaug'][c]['mean_diff']:+.3f}", flush=True)
    name = "j3_tcrzem_eegnet" + ("" if RECIPE == "default" else f"_{RECIPE}")
    name += "" if SEEDS == J.SEEDS else "_s" + "-".join(map(str, SEEDS))
    J.dump(res, name + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
