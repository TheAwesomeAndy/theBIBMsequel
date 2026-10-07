#!/usr/bin/env python3
"""Spherical-spline interpolation of missing electrodes (Perrin et al., 1989).

g(x) = 1/(4 pi) sum_{n=1}^{N} (2n+1) / (n (n+1))^m P_n(x), with m = 4 and N = 7
Legendre terms (the defaults used by common EEG toolboxes). For retained electrodes
F and missing electrodes M on the unit sphere, the interpolation matrix solves

    [G_FF + alpha I   1] [c ]   [v]
    [1^T              0] [c0] = [0]

and maps retained potentials v to missing ones as G_MF c + c0. alpha = 1e-5 regularizes
the solve. The matrix depends only on the montage and the missing set, never on data,
so it is computed once per (seed, fold) removal draw.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre


def sph_to_cart(theta_phi_deg):
    """BrainVision (theta, phi) in degrees -> unit vectors (x right, y anterior, z up)."""
    th = np.deg2rad(theta_phi_deg[:, 0]); ph = np.deg2rad(theta_phi_deg[:, 1])
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)


def _g(cosang, m=4, n_terms=7):
    n = np.arange(1, n_terms + 1)
    coef = np.concatenate([[0.0], (2 * n + 1) / (n * (n + 1)) ** m / (4 * np.pi)])
    return legendre.legval(np.clip(cosang, -1.0, 1.0), coef)


def interp_matrix(pos, missing, alpha=1e-5):
    """(len(missing), len(retained)) matrix mapping retained potentials to missing ones."""
    missing = np.asarray(missing, int)
    keep = np.setdiff1d(np.arange(len(pos)), missing)
    pf, pm = pos[keep], pos[missing]
    G = _g(pf @ pf.T) + alpha * np.eye(len(keep))
    A = np.block([[G, np.ones((len(keep), 1))], [np.ones((1, len(keep))), np.zeros((1, 1))]])
    Ainv = np.linalg.pinv(A)
    Gm = np.hstack([_g(pm @ pf.T), np.ones((len(missing), 1))])
    return (Gm @ Ainv)[:, : len(keep)], keep


def repair(X, pos, missing):
    """X (n, T, ch) with `missing` channels unknown -> copy with them interpolated."""
    out = X.copy()
    if len(missing) == 0:
        return out
    M, keep = interp_matrix(pos, missing)
    out[:, :, missing] = X[:, :, keep] @ M.T
    return out
