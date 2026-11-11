"""Fixed LIF reservoir, spike-bin codes, damage spreading, and threshold density.

The update is the one the BIBM code implements (camera-ready audit A1/A2):

    v <- (1 - beta) * v * (1 - z) + W_in x_t + W z
    z <- 1[v >= theta]
    v <- max(v - theta * z, 0)

One reservoir (256 units, one input) is shared by every electrode; each electrode trace
drives its own copy. With ``weight_seed=42`` and ``rho=0.9`` the weights are bit-identical
to BIBM (``tests/test_reservoir.py`` checks this against the BIBM reference code).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


def init_weights(n_res: int = C.N_RES, rho: float = C.RHO, seed: int = C.WEIGHT_SEED):
    """Xavier-uniform input and recurrent weights; W rescaled to spectral radius ``rho``."""
    rng = np.random.RandomState(seed)
    lin = np.sqrt(6.0 / (1 + n_res))
    W_in = rng.uniform(-lin, lin, (n_res, 1))
    lrec = np.sqrt(6.0 / (2 * n_res))
    W = rng.uniform(-lrec, lrec, (n_res, n_res))
    W = W * (rho / np.abs(np.linalg.eigvals(W)).max())
    return W_in, W


def mean_column_l1(W: np.ndarray) -> float:
    """Mean column sum of |W| after rescaling W to unit spectral radius (kappa-bar)."""
    r = np.abs(np.linalg.eigvals(W)).max()
    return float(np.abs(W / r).sum(axis=0).mean()) if r > 0 else 0.0


@dataclass
class ReservoirRun:
    counts: np.ndarray        # (n_obs, n_ch, n_res * n_bins) spike counts per unit and bin
    rate: float               # mean firing probability per unit and step inside the window
    p_theta: float | None     # density of pre-threshold membrane at theta (per unit potential)
    hist: np.ndarray | None   # membrane histogram counts inside the window
    hist_edges: np.ndarray | None


class LIFReservoir:
    def __init__(self, n_res: int = C.N_RES, beta: float = C.BETA, theta: float = C.THETA,
                 rho: float = C.RHO, weight_seed: int = C.WEIGHT_SEED):
        self.n_res, self.beta, self.theta, self.rho = n_res, beta, theta, rho
        self.weight_seed = weight_seed
        self.W_in, self.W = init_weights(n_res, rho, weight_seed)
        if rho == 0:
            self.W = np.zeros_like(self.W)

    def kappa_bar(self) -> float:
        """Column l1 norm at unit spectral radius, from the un-zeroed seed weights."""
        return mean_column_l1(init_weights(self.n_res, 1.0, self.weight_seed)[1])

    def _step(self, mem, spk, drive):
        mem = (1.0 - self.beta) * mem * (1.0 - spk) + self.W_in[:, :1] * drive[None, :] + self.W @ spk
        pre = mem
        spk = (mem >= self.theta).astype(np.float64)
        mem = np.maximum(mem - spk * self.theta, 0.0)
        return mem, spk, pre

    def encode(self, X: np.ndarray, window: tuple[int, int], n_bins: int = 6,
               membrane_stats: bool = False, h: float = 0.02) -> ReservoirRun:
        """Spike-bin code for every (observation, channel) drive.

        ``X`` is (n_obs, n_times, n_ch); ``window`` is [start, end) in samples. Bins have
        width ``(end - start) // n_bins``; samples after the last full bin are ignored.
        With ``membrane_stats`` the pre-threshold membrane inside the window is histogrammed
        and ``p_theta`` = fraction of values in [theta - h, theta + h) divided by 2h.
        """
        n_obs, T, n_ch = X.shape
        t0, t1 = window
        if not (0 <= t0 < t1 <= T):
            raise ValueError(f"window {window} outside signal of length {T}")
        binw = (t1 - t0) // n_bins
        if binw < 1:
            raise ValueError("window shorter than the number of bins")
        Cn = n_obs * n_ch
        S = np.transpose(X, (1, 0, 2)).reshape(T, Cn).astype(np.float64)
        mem = np.zeros((self.n_res, Cn))
        spk = np.zeros((self.n_res, Cn))
        counts = np.zeros((self.n_res, n_bins, Cn), dtype=np.float32)
        edges = np.linspace(0.0, 2.0 * self.theta + 1.0, 401) if membrane_stats else None
        hist = np.zeros(400) if membrane_stats else None
        near = 0
        total = 0
        for t in range(t1):
            mem, spk, pre = self._step(mem, spk, S[t])
            if t0 <= t < t0 + n_bins * binw:
                counts[:, (t - t0) // binw, :] += spk
                if membrane_stats:
                    hist += np.histogram(np.clip(pre, edges[0], edges[-1]), edges)[0]
                    near += int(((pre >= self.theta - h) & (pre < self.theta + h)).sum())
                    total += pre.size
        n_steps = n_bins * binw
        rate = float(counts.sum() / (self.n_res * Cn * n_steps))
        p_theta = near / (total * 2 * h) if membrane_stats else None
        codes = counts.reshape(self.n_res * n_bins, Cn).T.reshape(n_obs, n_ch, -1)
        return ReservoirRun(codes, rate, p_theta, hist, edges)

    def damage(self, X: np.ndarray, window: tuple[int, int], n_drive: int = 400,
               settle: int = 10, rng_seed: int = 0) -> dict:
        """Damage-spreading order parameter (BIBM definition) and its rate-normalized form.

        Two copies get the same drive; at ``window[0]`` one unit's spike is flipped in copy 2.
        Returns the mean normalized Hamming distance over steps more than ``settle`` after the
        flip, and that distance divided by 2r(1-r), the expected distance between unrelated
        spike vectors at the measured rate r.
        """
        n_obs, T, n_ch = X.shape
        t0, t1 = window
        rng = np.random.default_rng(rng_seed)
        cols = rng.choice(n_obs * n_ch, min(n_drive, n_obs * n_ch), replace=False)
        S = np.transpose(X, (1, 0, 2)).reshape(T, n_obs * n_ch)[:, cols].astype(np.float64)
        nd = S.shape[1]
        flip = rng.integers(0, self.n_res, nd)
        mem = [np.zeros((self.n_res, nd)) for _ in range(2)]
        spk = [np.zeros((self.n_res, nd)) for _ in range(2)]
        dist, rates = [], []
        for t in range(t1):
            for k in range(2):
                mem[k], spk[k], _ = self._step(mem[k], spk[k], S[t])
            if t == t0:
                spk[1][flip, np.arange(nd)] = 1.0 - spk[1][flip, np.arange(nd)]
            if t > t0 + settle:
                dist.append(np.abs(spk[0] - spk[1]).mean(axis=0))
                rates.append(spk[0].mean())
        d = float(np.mean(dist)) if dist else float("nan")
        r = float(np.mean(rates)) if rates else float("nan")
        ref = 2 * r * (1 - r)
        return {"damage": d, "rate": r, "damage_rate_normalized": d / ref if ref > 0 else float("nan")}


def branching_count(rho: float, p_theta: float, kappa_bar: float) -> float:
    """One-step branching count m(rho) ~ rho * p_theta * kappa_bar (BIBM Sec. II-B)."""
    return rho * p_theta * kappa_bar


def bsc1_from_bsc(codes: np.ndarray, n_res: int = C.N_RES, n_bins: int = 6) -> np.ndarray:
    """Collapse a spike-bin code to its single-bin (total count) version."""
    n_obs, n_ch, _ = codes.shape
    return codes.reshape(n_obs, n_ch, n_res, n_bins).sum(axis=3)
