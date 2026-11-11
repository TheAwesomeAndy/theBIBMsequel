"""Trained baselines: EEGNet-8,2 (with and without augmentation) and Riemannian tangent space.

EEGNet follows the BIBM camera-ready recipe exactly (bibm_baseline/analysis/
experiment_eegnet_signal.py): F1=8, D=2, F2=16, temporal kernel 64, dropout 0.25, Adam
lr 1e-3, 80 epochs, batch 64; augmentation zeroes 0-50% of channels in half the samples and
adds Gaussian noise with SD ~ U(0, 0.5); inputs are standardized per channel with
training statistics. Only the channel count, sample count and class count are now
parameters, so the network runs on every dataset.

The Riemannian baselines use ERP-specific covariances (pyriemann): ``ERPCovariances``
augments each trial with the class-mean prototypes of the training data before estimating
a shrunk covariance, and ``XdawnCovariances`` first projects onto xDAWN spatial filters.
Both map to the tangent space at the training mean and use the same balanced logistic
readout as the fixed encoders. Prototypes and filters use training labels only.
"""
from __future__ import annotations

import numpy as np

EPOCHS, BATCH, LR, DROPOUT = 80, 64, 1e-3, 0.25


# ------------------------------------------------------------------------------- EEGNet
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_eegnet(n_ch: int, n_t: int, n_cls: int, F1=8, D=2, F2=16, k=64, p=DROPOUT):
    torch, nn = _torch()

    class EEGNet(nn.Module):
        def __init__(self):
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

    return EEGNet()


def augment(xb, gen):
    """BIBM augmentation: p=0.5 zero 0-50% of channels; additive noise SD ~ U(0, 0.5)."""
    torch, _ = _torch()
    n, _, c, _ = xb.shape
    k = torch.randint(0, c // 2 + 1, (n, 1), generator=gen)
    k = k * (torch.rand(n, 1, generator=gen) < 0.5)
    rank = torch.argsort(torch.rand(n, c, generator=gen), dim=1).argsort(dim=1)
    keep = (rank >= k).float()[:, None, :, None]
    sd = torch.rand(n, 1, 1, 1, generator=gen) * 0.5
    return xb * keep + sd * torch.randn(xb.shape, generator=gen)


class EEGNetClassifier:
    """fit / predict_proba wrapper. X is (n, T, ch) as everywhere else in the package."""

    def __init__(self, seed: int, aug: bool = False, epochs: int = EPOCHS, threads: int = 1):
        self.seed, self.aug, self.epochs, self.threads = seed, aug, epochs, threads

    def _input(self, X):
        torch, _ = _torch()
        Z = ((X - self.mu) / self.sd).transpose(0, 2, 1)[:, None, :, :]
        return torch.tensor(Z, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch, nn = _torch()
        torch.set_num_threads(self.threads)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu = X.mean(axis=(0, 1))
        self.sd = X.std(axis=(0, 1)) + 1e-8
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self.model = make_eegnet(X.shape[2], X.shape[1], len(self.classes_))
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        Xt, Yt = self._input(X), torch.tensor(yi, dtype=torch.long)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i + BATCH]
                xb = augment(Xt[idx], gen) if self.aug else Xt[idx]
                opt.zero_grad()
                lossf(self.model(xb), Yt[idx]).backward()
                opt.step()
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, _ = _torch()
        with torch.no_grad():
            return torch.softmax(self.model(self._input(X)), 1).numpy()


# --------------------------------------------------------------------------- Riemannian
def riemann_pipeline(kind: str = "erpcov", seed: int = 0, nfilter: int = 4):
    """Tangent-space pipeline on (n, ch, T) inputs. ``kind``: 'erpcov' or 'xdawn'."""
    from pyriemann.estimation import ERPCovariances, XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cov = (ERPCovariances(estimator="lwf") if kind == "erpcov"
           else XdawnCovariances(nfilter=nfilter, estimator="lwf"))
    return make_pipeline(cov, TangentSpace(metric="riemann"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                            solver="lbfgs", random_state=seed))


class RiemannClassifier:
    """Same interface as EEGNetClassifier: X is (n, T, ch)."""

    def __init__(self, kind: str = "erpcov", seed: int = 0):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        self.pipe = riemann_pipeline(self.kind, self.seed).fit(X.transpose(0, 2, 1), y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X.transpose(0, 2, 1))
