"""Channel fill rules.

Feature-level fills act on per-channel feature blocks of the TEST observations after
extraction (the BIBM feature-coordinate experiment). Signal-level fills act on the test
signal before extraction (removed traces are zeroed or interpolated, then every encoder
recomputes its features).

Feature-level rules (``fill_blocks``):
    zero     dropped block set to 0                       (not invariant to re-centering)
    mean     dropped block set to its training mean       (invariant, Proposition 1)
    knn      sklearn KNNImputer(k=10) fitted on training observations
    spatial  mean of the same observation's retained channels
    spline   spherical-spline interpolation per feature index (needs electrode positions)

``refit`` (train a readout on the retained channels only) is handled in the protocol,
because it changes the training side, not the test features.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre
from sklearn.impute import KNNImputer

FEATURE_FILLS = ("zero", "mean", "knn", "spatial", "spline")


def fill_blocks(Bte: np.ndarray, Btr: np.ndarray, drop: np.ndarray, rule: str,
                positions: np.ndarray | None = None, k: int = 10) -> np.ndarray:
    """Return a copy of test blocks (n_te, n_ch, f) with channels ``drop`` filled by ``rule``."""
    out = Bte.astype(np.float64, copy=True)
    if len(drop) == 0:
        return out
    n_te, n_ch, f = out.shape
    keep = np.setdiff1d(np.arange(n_ch), drop)
    if rule == "zero":
        out[:, drop, :] = 0.0
    elif rule == "mean":
        out[:, drop, :] = Btr[:, drop, :].mean(axis=0)
    elif rule == "knn":
        flat = out.reshape(n_te, -1)
        mask = np.zeros((n_ch, f), bool)
        mask[drop, :] = True
        flat[:, mask.reshape(-1)] = np.nan
        imp = KNNImputer(n_neighbors=k).fit(Btr.reshape(len(Btr), -1))
        out = imp.transform(flat).reshape(n_te, n_ch, f)
    elif rule == "spatial":
        out[:, drop, :] = out[:, keep, :].mean(axis=1, keepdims=True)
    elif rule == "spline":
        if positions is None:
            raise ValueError("spline fill needs electrode positions")
        M = spherical_spline_matrix(positions[keep], positions[drop])    # (n_drop, n_keep)
        out[:, drop, :] = np.einsum("dk,nkf->ndf", M, out[:, keep, :])
    else:
        raise ValueError(f"unknown fill rule {rule!r}")
    return out


def _g(cosang: np.ndarray, m: int = 4, n_terms: int = 50) -> np.ndarray:
    """Spherical-spline kernel g_m(x) = 1/(4 pi) sum_n (2n+1)/(n(n+1))^m P_n(x)."""
    n = np.arange(1, n_terms + 1)
    coef = np.zeros(n_terms + 1)
    coef[1:] = (2 * n + 1) / (n * (n + 1)) ** m
    return legendre.legval(np.clip(cosang, -1.0, 1.0), coef) / (4 * np.pi)


def spherical_spline_matrix(pos_from: np.ndarray, pos_to: np.ndarray, m: int = 4,
                            lam: float = 1e-5, n_terms: int = 50) -> np.ndarray:
    """Linear map from retained-electrode values to missing-electrode values (Perrin 1989).

    Positions are (n, 3) and are projected to the unit sphere. Solves
    [G + lam I, 1; 1^T, 0] [c; c0] = [v; 0] and evaluates g(to, from) c + c0.
    """
    a = pos_from / np.linalg.norm(pos_from, axis=1, keepdims=True)
    b = pos_to / np.linalg.norm(pos_to, axis=1, keepdims=True)
    n = len(a)
    G = _g(a @ a.T, m, n_terms) + lam * np.eye(n)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    Ainv = np.linalg.pinv(A)[:, :n]           # maps v -> [c; c0]
    Gt = np.hstack([_g(b @ a.T, m, n_terms), np.ones((len(b), 1))])
    return Gt @ Ainv


def interpolate_signal(X: np.ndarray, drop: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Signal-level spherical-spline repair of removed electrodes. X is (n, T, ch)."""
    if len(drop) == 0:
        return X.copy()
    keep = np.setdiff1d(np.arange(X.shape[2]), drop)
    M = spherical_spline_matrix(positions[keep], positions[drop])
    out = X.copy()
    out[:, :, drop] = np.einsum("dk,ntk->ntd", M, X[:, :, keep])
    return out
