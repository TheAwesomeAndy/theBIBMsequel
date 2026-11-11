"""Checks for the Phase 3 trained baselines (synthetic data, a few epochs)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyriemann")

from sequel import trained  # noqa: E402


def _toy(n=60, T=64, ch=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(0, 1, (n, T, ch))
    X[y == 1, 20:40, :3] += 1.5                     # class-specific deflection on 3 channels
    return X, y


@pytest.mark.parametrize("n_ch,n_t,n_cls", [(34, 205, 3), (30, 205, 2), (8, 64, 2)])
def test_eegnet_shapes(n_ch, n_t, n_cls):
    net = trained.make_eegnet(n_ch, n_t, n_cls)
    assert net(torch.zeros(2, 1, n_ch, n_t)).shape == (2, n_cls)


def test_augment_keeps_shape_and_zeroes_channels():
    gen = torch.Generator().manual_seed(0)
    xb = torch.ones(200, 1, 10, 20)
    out = trained.augment(xb, gen)
    assert out.shape == xb.shape
    zeroed = ((out - torch.zeros_like(out)).abs().mean(dim=(1, 3)) < 0.6).float().mean()
    assert 0.0 < zeroed < 0.5                       # some channels removed, never most


def test_eegnet_learns_and_is_deterministic():
    X, y = _toy()
    a = trained.EEGNetClassifier(seed=1, epochs=30).fit(X, y).predict_proba(X)
    b = trained.EEGNetClassifier(seed=1, epochs=30).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.mean(a.argmax(1) == y) > 0.8


@pytest.mark.parametrize("kind", ["erpcov", "xdawn"])
def test_riemann_learns_and_tolerates_zeroed_channel(kind):
    X, y = _toy(seed=2)
    clf = trained.RiemannClassifier(kind).fit(X, y)
    assert np.mean(clf.predict_proba(X).argmax(1) == y) > 0.8
    Xz = X.copy()
    Xz[:, :, 5] = 0.0                               # a removed electrode must not break SPD estimation
    p = clf.predict_proba(Xz)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
