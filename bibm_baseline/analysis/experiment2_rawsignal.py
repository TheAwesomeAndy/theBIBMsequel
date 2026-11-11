#!/usr/bin/env python3
"""Experiment 2: signal-level channel removal, every representation recomputed
from the corrupted signal (distinct from the Experiment 1 feature-coordinate
audit, where feature blocks are suppressed after extraction).

At the signal level a removed electrode contributes a zero trace, which yields
zero band-power, zero ERP-window amplitude, and a zero BSC6 block (a zero drive
from a zero reservoir state produces no spikes). The scientific difference from
Experiment 1 is the reservoir: here the zeroed channel enters BEFORE the PCA
mixing, and PCA is refit on the clean training fold only, so the effect
propagates across all 64 components instead of staying in one channel block.

Protocol matches the source: StratifiedGroupKFold(5) x seeds 42-46, train-only
StandardScaler + balanced L2 logreg; features fit on the clean train fold and
evaluated on the recomputed corrupted test fold. Per-subject out-of-fold
predictions (seed-aggregated) feed a subject-level bootstrap (211 subjects,
3 conditions each). EEGNet's raw-signal numbers are read from the existing
aggregate outputs (it ingests the raw signal directly).

Reads the restricted SHAPE pickle locally; writes only aggregate JSON.
"""
from __future__ import annotations
import json, os, pickle
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

N_RES, BETA, THETA, SEED = 256, 0.05, 0.5, 42
BSC_N_BINS, T_START, T_END, PCA_K = 6, 10, 70, 64
SEEDS = [42, 43, 44, 45, 46]
FS = 256
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
# Windows are in sample-index milliseconds (sample / 256 Hz), measured from the FIRST sample of
# X_ds, which is 200 ms before stimulus onset (X_ds == raw[0:1024:4]; onset at raw sample 205,
# X_ds sample 51). Relative to onset they span about -120..0, 50..250 and 250..600 ms.
ERP_WINDOWS_MS = [(80, 200), (250, 450), (450, 800)]
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_BOOT = 4000
PKL = os.environ.get("ARSPI_SHAPE_FEATURES",
                     "/home/user/operational-distinctness-paper/data/shape_features_211.pkl")


def init_weights():
    rng = np.random.RandomState(SEED)
    lin = np.sqrt(6.0 / (1 + N_RES)); W_in = rng.uniform(-lin, lin, (N_RES, 1))
    lrec = np.sqrt(6.0 / (2 * N_RES)); W_rec = rng.uniform(-lrec, lrec, (N_RES, N_RES))
    W_rec *= 0.9 / np.abs(np.linalg.eigvals(W_rec)).max()
    return W_in, W_rec


def bsc6_all(X_ds):
    """Clean BSC6 for every (obs,channel): returns (N,34,1536)."""
    N, T, n_ch = X_ds.shape; C = N * n_ch
    S = np.transpose(X_ds, (1, 0, 2)).reshape(T, C).astype(np.float64)
    W_in, W_rec = init_weights(); win0 = W_in[:, 0][:, None]
    mem = np.zeros((N_RES, C)); spk = np.zeros((N_RES, C))
    counts = np.zeros((N_RES, BSC_N_BINS, C), dtype=np.float32)
    binw = (T_END - T_START) // BSC_N_BINS
    for t in range(T):
        I = win0 * S[t][None, :] + W_rec @ spk
        mem = (1.0 - BETA) * mem * (1.0 - spk) + I
        spk = (mem >= THETA).astype(np.float64)
        mem = np.maximum(mem - spk * THETA, 0.0)
        if T_START <= t < T_END:
            b = (t - T_START) // binw
            if b < BSC_N_BINS:
                counts[:, b, :] += spk
    bsc = counts.reshape(N_RES * BSC_N_BINS, C).T  # (C,1536), row=obs*34+ch
    return bsc.reshape(N, n_ch, N_RES * BSC_N_BINS)


def bandpower_all(X_ds):
    N, T, n_ch = X_ds.shape
    out = np.zeros((N, n_ch, 5))
    for i in range(N):
        for ch in range(n_ch):
            f, p = welch(X_ds[i, :, ch], fs=FS, nperseg=min(T, 256))
            for bi, (lo, hi) in enumerate(BANDS):
                m = (f >= lo) & (f <= hi)
                if m.any():
                    out[i, ch, bi] = np.trapezoid(p[m], f[m])
    return out


def erp_all(X_ds):
    N, T, n_ch = X_ds.shape
    out = np.zeros((N, n_ch, len(ERP_WINDOWS_MS)))
    for w, (a, b) in enumerate(ERP_WINDOWS_MS):
        out[:, :, w] = X_ds[:, int(round(a/1000*FS)):int(round(b/1000*FS)), :].mean(axis=1)
    return out


def zero_channels(feat_perch, drop_ch):
    out = feat_perch.copy(); out[:, drop_ch, :] = 0.0; return out


def oof_predictions(y, groups, frac,
                    band_perch, erp_perch, bsc_perch):
    """Return dict name -> (N,) seed-aggregated OOF predictions at dropout frac,
    recomputing each representation from the channel-zeroed signal."""
    N = len(y); classes = np.unique(y); n_ch = band_perch.shape[1]
    acc = {n: np.zeros((N, classes.size)) for n in ("Band-power", "ERP-window", "Reservoir")}
    seen = np.zeros(N)
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros((N, 1)), y, groups=groups)):
            rng = np.random.default_rng(1000 * seed + fold)
            drop = rng.choice(n_ch, size=int(round(frac * n_ch)), replace=False) if frac > 0 else np.array([], int)
            # clean-train / corrupted-test for each representation
            for name, perch in (("Band-power", band_perch), ("ERP-window", erp_perch)):
                Xtr = perch[tr].reshape(len(tr), -1)
                Xte = zero_channels(perch[te], drop).reshape(len(te), -1)
                _fit_pred(Xtr, y[tr], Xte, seed, acc[name], te, classes)
            # reservoir: PCA refit on clean train BSC6, transform corrupted test BSC6
            pca = PCA(n_components=PCA_K, random_state=SEED)
            Etr = pca.fit_transform(bsc_perch[tr].reshape(-1, bsc_perch.shape[2])).reshape(len(tr), n_ch * PCA_K)
            bsc_te = zero_channels(bsc_perch[te], drop)
            Ete = pca.transform(bsc_te.reshape(-1, bsc_perch.shape[2])).reshape(len(te), n_ch * PCA_K)
            _fit_pred(Etr, y[tr], Ete, seed, acc["Reservoir"], te, classes)
            seen[te] += 1
    return {n: classes[np.argmax(acc[n] / np.maximum(seen[:, None], 1), axis=1)] for n in acc}


def _fit_pred(Xtr, ytr, Xte, seed, acc, te, classes):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="lbfgs", random_state=seed).fit(sc.transform(Xtr), ytr)
    acc[te] += clf.predict_proba(sc.transform(Xte))


def subj_boot(y, groups, pred_map, n_boot=N_BOOT, seed=7):
    rng = np.random.default_rng(seed); sid = np.unique(groups)
    s2i = {s: np.where(groups == s)[0] for s in sid}
    names = list(pred_map)
    pt = {n: balanced_accuracy_score(y, pred_map[n]) for n in names}
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        idx = np.concatenate([s2i[s] for s in rng.choice(sid, sid.size, replace=True)])
        for n in names:
            boot[n].append(balanced_accuracy_score(y[idx], pred_map[n][idx]))
    ci = {n: [float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))] for n in names}
    return pt, ci


def main():
    d = pickle.load(open(PKL, "rb"))
    X_ds = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    print("[exp2] precomputing clean features (band-power, ERP, BSC6)...")
    band = bandpower_all(X_ds); erp = erp_all(X_ds); bsc = bsc6_all(X_ds)
    res = {"protocol": "Experiment 2 (signal-level channel removal; features recomputed "
           "from corrupted signal; reservoir PCA refit on clean train). "
           "StratifiedGroupKFold(5) x seeds 42-46; subject-level bootstrap n_boot=%d" % N_BOOT,
           "note_eegnet": "EEGNet ingests the raw signal; its raw-dropout BA is in eegnet.json "
                          "(clean 0.711, 30% dropout 0.554; +aug 0.674).",
           "BA": {}}
    # clean (0%) validation + curve
    for frac in [0.0] + LEVELS:
        pm = oof_predictions(y, g, frac, band, erp, bsc)
        pt, ci = subj_boot(y, g, pm)
        res["BA"][f"{frac:.1f}"] = {n: {"BA": pt[n], "ci95": ci[n]} for n in pt}
        tag = "clean" if frac == 0 else f"{int(frac*100)}%"
        print(f"[exp2 {tag}] " + " | ".join(f"{n[:4]}={pt[n]:.3f}[{ci[n][0]:.3f},{ci[n][1]:.3f}]" for n in pt))
    out = Path(__file__).resolve().parents[1] / "outputs" / "aggregate" / "experiment2_rawsignal.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[out] wrote {out}")
    print(f"[validate] reservoir clean (per-fold PCA) BA={res['BA']['0.0']['Reservoir']['BA']:.4f} "
          f"(source fold-local control 0.465)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
