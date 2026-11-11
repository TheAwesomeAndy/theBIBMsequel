"""Journal extension of the BIBM 2026 paper (Lane, Tang, Nelson).

Modules:
    config      timing constants and window definitions (relative to stimulus onset)
    data        SHAPE loaders and preprocessing (legacy BIBM path and corrected path)
    reservoir   fixed LIF reservoir, spike-bin codes, damage spreading, threshold density
    features    band power, ERP windows, matched amplitude bins, fold-local block PCA
    fills       channel fill rules (zero, mean, kNN, spatial, spherical spline, refit)
    protocol    subject-grouped CV, pooled out-of-fold probabilities, subject bootstrap,
                within-subject permutation test
    theory      zero-fill logit shift (Proposition 1) and its predicted accuracy effect
"""

__all__ = ["config", "data", "reservoir", "features", "fills", "protocol", "theory"]
