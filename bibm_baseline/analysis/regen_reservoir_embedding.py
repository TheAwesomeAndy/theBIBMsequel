#!/usr/bin/env python3
"""Faithfully regenerate the reservoir BSC6/PCA-64 embedding (E stream) from the
downsampled EEG (X_ds), because the SHAPE feature pickle ships this block zeroed.

Exact replica of dissoAdventureExperiments/experiments/ch5_4class/
ch5_4class_01_feature_extraction.py:
  * LIFReservoir: N=256, beta=0.05, theta=0.5, seed=42, Xavier-uniform W_in/W_rec,
    spectral radius 0.9, multiplicative reset, membrane floor at 0.
  * BSC6: spikes[10:70] partitioned into 6 bins of 10 -> 256*6 = 1536 per channel.
  * Global PCA-64 fit on the pooled (N_obs*34, 1536) matrix -> (N_obs, 34, 64).

Vectorized over all 21,522 (obs,channel) reservoir drives at once.
Writes the embedding to a LOCAL, git-ignored .npy (never committed: it is a
subject-level derived feature of restricted data). Validates clean BA against
the published 0.463 under the source subject-grouped protocol.
"""
from __future__ import annotations
import os, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

N_RES, BETA, THETA, SEED = 256, 0.05, 0.5, 42
BSC_N_BINS, T_START, T_END, PCA_K = 6, 10, 70, 64
SEEDS = [42, 43, 44, 45, 46]

PKL = os.environ.get("ARSPI_SHAPE_FEATURES",
                     "/home/user/operational-distinctness-paper/data/shape_features_211.pkl")
OUT = Path(os.environ.get("ARSPI_RESERVOIR_NPY", "/tmp/reservoir_bsc6_pca_211.npy"))


def init_weights():
    rng = np.random.RandomState(SEED)
    lin = np.sqrt(6.0 / (1 + N_RES))
    W_in = rng.uniform(-lin, lin, (N_RES, 1))
    lrec = np.sqrt(6.0 / (N_RES + N_RES))
    W_rec = rng.uniform(-lrec, lrec, (N_RES, N_RES))
    sr = np.abs(np.linalg.eigvals(W_rec)).max()
    if sr > 0:
        W_rec *= 0.9 / sr
    return W_in, W_rec


def regen(X_ds):
    """X_ds: (N,256,34) z-scored EEG. Returns (N,34,64) embedding + fitted PCA."""
    N, T, n_ch = X_ds.shape
    C = N * n_ch
    # column c = obs*34 + ch  (matches bsc6_raw.reshape(-1,1536) row order)
    S = np.transpose(X_ds, (1, 0, 2)).reshape(T, C).astype(np.float64)  # (T, C)
    W_in, W_rec = init_weights()
    win0 = W_in[:, 0][:, None]                      # (N_RES,1)
    mem = np.zeros((N_RES, C)); spk_prev = np.zeros((N_RES, C))
    counts = np.zeros((N_RES, BSC_N_BINS, C), dtype=np.float32)
    for t in range(T):
        I_tot = win0 * S[t][None, :] + W_rec @ spk_prev
        mem = (1.0 - BETA) * mem * (1.0 - spk_prev) + I_tot
        spk = (mem >= THETA).astype(np.float64)
        mem = np.maximum(mem - spk * THETA, 0.0)
        if T_START <= t < T_END:
            b = (t - T_START) // ((T_END - T_START) // BSC_N_BINS)
            if b < BSC_N_BINS:
                counts[:, b, :] += spk
        spk_prev = spk
    # per column 1536 vector ordered neuron*6 + bin  (extract_bsc flatten order)
    bsc = counts.reshape(N_RES * BSC_N_BINS, C).T           # (C, 1536)
    pca = PCA(n_components=PCA_K, random_state=SEED)
    emb = pca.fit_transform(bsc).reshape(N, n_ch, PCA_K)    # (N,34,64)
    return emb, pca


def validate(emb, y, groups):
    X = emb.reshape(len(y), -1)
    bas = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups=groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     solver="lbfgs", random_state=seed).fit(sc.transform(X[tr]), y[tr])
            bas.append(balanced_accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(bas))


def main():
    d = pickle.load(open(PKL, "rb"))
    X_ds = np.asarray(d["X_ds"], float); y = np.asarray(d["y"]); g = np.asarray(d["subjects"])
    print(f"[regen] X_ds {X_ds.shape} zscored? per-ch mean~{X_ds.mean():.3f} std~{X_ds.std():.3f}")
    emb, pca = regen(X_ds)
    print(f"[regen] embedding {emb.shape} | mean {emb.mean():.4f} std {emb.std():.4f} "
          f"| pca var explained {pca.explained_variance_ratio_[:3].round(3).tolist()}")
    ba = validate(emb, y, g)
    print(f"[validate] regenerated reservoir clean BA = {ba:.4f}  (published 0.463)")
    ok = abs(ba - 0.463) <= 0.02
    np.save(OUT, emb)
    print(f"[out] saved embedding -> {OUT}  (git-ignored; NOT committed)")
    print("[validate] MATCH" if ok else "[validate] MISMATCH — do not use until reconciled")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
