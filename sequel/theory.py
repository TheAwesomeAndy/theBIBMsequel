"""Zero-fill logit shift and its predicted accuracy effect.

Setting (BIBM Proposition 1). The readout sees z = D_sigma^{-1}(x - mu), with mu, sigma fitted
on training subjects, and class logits l = W z + b. Let S be the dropped coordinates.

* Zero fill and mean fill differ by a fixed shift of the standardized input,
  z_S^zero - z_S^mean = -D_{sigma,S}^{-1} mu_S, hence by a fixed logit shift
      delta = -W_{:,S} (mu_S / sigma_S),
  identical for every test observation. Only its class-centered part
  delta - mean(delta) changes predictions, because softmax ignores a common offset.

* Proposition 3 (proved in docs/THEORY.md). Scale the encoder's offset so that
  delta = t u with u fixed and a unique largest centered coordinate k*. Mean-fill balanced
  accuracy does not depend on t. As t -> infinity, every zero-filled observation is
  assigned to k*, so zero-fill balanced accuracy tends to the chance level 1/K.

``fill_bias_index`` is a training-side predictor of how much zero fill will move a model:
the spread of the centered logit shift divided by the median top-two logit margin on the
training data. It uses no test labels.
"""
from __future__ import annotations

import numpy as np


def _coef(clf) -> np.ndarray:
    W = np.asarray(clf.coef_)
    return np.vstack([np.zeros_like(W), W]) if W.shape[0] == 1 else W


def zero_fill_shift(scaler, clf, cols: np.ndarray) -> np.ndarray:
    """Class-logit shift from replacing mean fill by zero fill on ``cols`` (flat indices)."""
    cols = np.asarray(cols, int)
    W = _coef(clf)
    return -(W[:, cols] @ (scaler.mean_[cols] / scaler.scale_[cols]))


def centered_spread(delta: np.ndarray) -> float:
    d = delta - delta.mean()
    return float(d.max() - d.min())


def logits(scaler, clf, X: np.ndarray) -> np.ndarray:
    W = _coef(clf)
    b = np.asarray(clf.intercept_)
    b = np.r_[0.0, b] if b.size == 1 else b
    return scaler.transform(X) @ W.T + b


def median_margin(scaler, clf, X: np.ndarray) -> float:
    L = np.sort(logits(scaler, clf, X), axis=1)
    return float(np.median(L[:, -1] - L[:, -2]))


def fill_bias_index(scaler, clf, cols: np.ndarray, Xtr: np.ndarray) -> float:
    return centered_spread(zero_fill_shift(scaler, clf, cols)) / max(median_margin(scaler, clf, Xtr), 1e-12)


def non_centeredness(scaler) -> float:
    """Mean |mu_i| / sigma_i over features: the encoder-level quantity in Fig. 3 of BIBM."""
    return float(np.mean(np.abs(scaler.mean_) / scaler.scale_))


def block_cols(drop: np.ndarray, n_feat: int) -> np.ndarray:
    """Flat column indices of dropped channel blocks in a (n_ch * n_feat) layout."""
    drop = np.asarray(drop, int)
    return (drop[:, None] * n_feat + np.arange(n_feat)[None, :]).reshape(-1)
