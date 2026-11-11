"""Data-independent checks. Run: python -m pytest -q tests

They prove implementation contracts (exact BIBM reproduction, preprocessing boundaries,
the fill propositions, protocol invariants). They say nothing about empirical results.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sequel import config as C
from sequel import data, features, fills, protocol, reservoir, theory
from tests import bibm_reference as REF
from tests.synthetic import electrode_positions, raw_epochs


@pytest.fixture(scope="module")
def raw():
    return raw_epochs(n_subj=12)


@pytest.fixture(scope="module")
def legacy(raw):
    X, y, g = raw
    return data.preprocess(X, y, g, decimation="legacy", norm="epoch_z")


# ------------------------------------------------------------------------------- data
def test_legacy_matches_bibm_definition(raw):
    X, y, g = raw
    ep = data.preprocess(X, y, g, decimation="legacy", norm="epoch_z")
    ref = X[:, 0:1024:4, :]
    ref = (ref - ref.mean(1, keepdims=True)) / ref.std(1, keepdims=True)
    assert ep.X.shape == (len(y), 256, 34)
    np.testing.assert_allclose(ep.X, ref, atol=1e-12)
    assert ep.onset == C.ONSET


def test_fir_decimation_alignment_and_antialias():
    t = np.arange(1229) / 1024.0
    tone = np.sin(2 * np.pi * 200.0 * t)               # above the 128 Hz Nyquist of 256 Hz
    raw = np.repeat(tone[None, :, None], 2, axis=2)
    naive, _ = data.decimate_raw(raw, "legacy")
    fir, times = data.decimate_raw(raw, "fir")
    assert fir.shape[1] == 307
    assert abs(times[C.ONSET]) < 1e-9                  # onset lands exactly on sample 51
    core = slice(20, -20)                              # ignore filter edges
    assert naive[0, core, 0].std() > 0.5               # 200 Hz aliases to 56 Hz unfiltered
    assert fir[0, core, 0].std() < 0.01 * naive[0, core, 0].std()   # > 40 dB attenuation


def test_baseline_z_prestimulus_independent_of_poststimulus(raw):
    X, y, g = raw
    a = data.preprocess(X, y, g, "fir", "baseline_z").X
    X2 = X.copy()
    X2[:, 400:, :] *= 5.0                              # change only late post-stimulus samples
    b = data.preprocess(X2, y, g, "fir", "baseline_z").X
    pre = slice(0, C.ONSET - 10)                       # stay clear of the FIR support
    np.testing.assert_allclose(a[:, pre], b[:, pre], atol=1e-9)
    e1 = data.preprocess(X, y, g, "fir", "epoch_z").X
    e2 = data.preprocess(X2, y, g, "fir", "epoch_z").X
    assert np.abs(e1[:, pre] - e2[:, pre]).max() > 1e-2   # epoch_z leaks post into pre


# -------------------------------------------------------------------------- reservoir
@pytest.mark.parametrize("rho", [0.0, 0.9, 1.5])
def test_reservoir_bit_identical_to_bibm(legacy, rho):
    Xs = legacy.X[:6]
    ref_codes, ref_rate = REF.bsc6(Xs, rho)
    run = reservoir.LIFReservoir(rho=rho).encode(Xs, C.LEGACY_RES_WINDOW, n_bins=6)
    np.testing.assert_array_equal(run.counts, ref_codes)
    assert run.rate == pytest.approx(ref_rate, abs=1e-12)


def test_window_conversion_matches_bibm():
    assert C.window_samples(C.RES_WINDOWS_REL_MS["bibm"]) == C.LEGACY_RES_WINDOW


def test_bsc1_is_total_count(legacy):
    run = reservoir.LIFReservoir().encode(legacy.X[:3], (10, 70), 6)
    b1 = reservoir.bsc1_from_bsc(run.counts)
    np.testing.assert_allclose(b1.sum(axis=2), run.counts.sum(axis=2))
    assert b1.shape == (3, 34, C.N_RES)


def test_membrane_density_and_branching(legacy):
    res = reservoir.LIFReservoir()
    run = res.encode(legacy.X[:4], (10, 70), 6, membrane_stats=True)
    assert run.p_theta is not None and run.p_theta >= 0
    assert run.hist.sum() == 60 * C.N_RES * 4 * 34
    kb = res.kappa_bar()
    assert kb == pytest.approx(13.4, abs=0.1)           # value quoted in the BIBM paper
    assert reservoir.branching_count(0.9, run.p_theta, kb) == pytest.approx(0.9 * run.p_theta * kb)


def test_damage_zero_without_recurrence(legacy):
    # Without W a flip cannot reach other units, but it does change the flipped unit's own
    # reset, so damage is confined to one unit: at most 1/N_RES of the spike vector.
    d = reservoir.LIFReservoir(rho=0.0).damage(legacy.X, (10, 70), n_drive=50)
    assert 0.0 <= d["damage"] <= 1.0 / C.N_RES
    d9 = reservoir.LIFReservoir(rho=0.9).damage(legacy.X, (10, 70), n_drive=50)
    assert d9["damage"] > 1.0 / C.N_RES                 # with recurrence it spreads


# --------------------------------------------------------------------------- features
def test_bandpower_matches_bibm(legacy):
    np.testing.assert_allclose(features.bandpower(legacy.X[:5]), REF.bandpower_all(legacy.X[:5]),
                               rtol=1e-10, atol=1e-14)


def test_erp_windows_relative(legacy):
    f = features.erp_windows(legacy.X, legacy.times_ms)
    assert f.shape == (legacy.X.shape[0], 34, 3)


def test_amplitude_bins_match_reservoir_bins(legacy):
    A = features.amplitude_bins(legacy.X, (10, 70), 6)
    np.testing.assert_allclose(A[:, :, 0], legacy.X[:, 10:20, :].mean(axis=1))
    np.testing.assert_allclose(A[:, :, 5], legacy.X[:, 60:70, :].mean(axis=1))


# ------------------------------------------------------------------------------ fills
def _fit(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=2000).fit(sc.transform(X), y)


def test_proposition1_mean_fill_invariant_zero_fill_not():
    rng = np.random.default_rng(0)
    n, n_ch, f = 120, 6, 4
    y = np.repeat([0, 1, 2], n // 3)
    B = rng.normal(0, 1, (n, n_ch, f)) + y[:, None, None] * 0.6
    drop = np.array([1, 4])
    preds = {}
    for b in (0.0, 5.0):                               # constant re-centering x -> x - b
        Bt = B - b
        tr, te = np.arange(0, n, 2), np.arange(1, n, 2)
        sc, clf = _fit(Bt[tr].reshape(len(tr), -1), y[tr])
        for rule in ("mean", "zero"):
            Xte = fills.fill_blocks(Bt[te], Bt[tr], drop, rule).reshape(len(te), -1)
            preds[(b, rule)] = clf.predict_proba(sc.transform(Xte))
    np.testing.assert_allclose(preds[(0.0, "mean")], preds[(5.0, "mean")], atol=1e-6)
    assert np.abs(preds[(0.0, "zero")] - preds[(5.0, "zero")]).max() > 1e-2


def test_zero_fill_shift_formula_exact():
    rng = np.random.default_rng(1)
    n, n_ch, f = 90, 5, 3
    y = np.repeat([0, 1, 2], n // 3)
    B = rng.normal(2.0, 1.0, (n, n_ch, f)) + y[:, None, None]
    drop = np.array([0, 3])
    sc, clf = _fit(B.reshape(n, -1), y)
    lz = theory.logits(sc, clf, fills.fill_blocks(B, B, drop, "zero").reshape(n, -1))
    lm = theory.logits(sc, clf, fills.fill_blocks(B, B, drop, "mean").reshape(n, -1))
    delta = theory.zero_fill_shift(sc, clf, theory.block_cols(drop, f))
    np.testing.assert_allclose(lz - lm, np.broadcast_to(delta, lz.shape), atol=1e-9)


def test_proposition3_zero_fill_tends_to_chance():
    rng = np.random.default_rng(2)
    n, n_ch, f = 300, 6, 3
    y = np.repeat([0, 1, 2], n // 3)
    B = rng.normal(0, 1, (n, n_ch, f))
    B[:, :, 0] += (y[:, None] - 1) * 1.5
    tr, te = np.arange(0, n, 2), np.arange(1, n, 2)
    drop = np.array([0, 1, 2])
    ba = {}
    for b in (0.0, 1e4):
        Bt = B + b
        sc, clf = _fit(Bt[tr].reshape(len(tr), -1), y[tr])
        for rule in ("mean", "zero"):
            p = clf.predict(sc.transform(fills.fill_blocks(Bt[te], Bt[tr], drop, rule).reshape(len(te), -1)))
            ba[(b, rule)] = np.mean([np.mean(p[y[te] == k] == k) for k in range(3)])
    assert ba[(0.0, "mean")] == pytest.approx(ba[(1e4, "mean")], abs=1e-9)
    assert ba[(1e4, "zero")] == pytest.approx(1 / 3, abs=1e-9)


def test_spherical_spline_reproduces_smooth_field():
    pos = electrode_positions(34)
    field = 1.0 + pos[:, 0] + 0.5 * pos[:, 1]            # smooth (degree-1) scalp field
    drop = np.array([3, 10, 22])
    keep = np.setdiff1d(np.arange(34), drop)
    M = fills.spherical_spline_matrix(pos[keep], pos[drop])
    assert np.abs(M @ field[keep] - field[drop]).max() < 0.05
    Mself = fills.spherical_spline_matrix(pos[keep], pos[keep], lam=0.0)
    np.testing.assert_allclose(Mself @ field[keep], field[keep], atol=1e-6)


def test_spatial_and_knn_fill_shapes():
    rng = np.random.default_rng(3)
    Btr, Bte = rng.normal(size=(40, 8, 3)), rng.normal(size=(10, 8, 3))
    drop = np.array([2, 5])
    sp = fills.fill_blocks(Bte, Btr, drop, "spatial")
    np.testing.assert_allclose(sp[:, 2, :], Bte[:, np.setdiff1d(np.arange(8), drop), :].mean(1))
    kn = fills.fill_blocks(Bte, Btr, drop, "knn")
    assert np.isfinite(kn).all()
    np.testing.assert_array_equal(kn[:, 0, :], Bte[:, 0, :])


# --------------------------------------------------------------------------- protocol
def test_splits_never_share_subjects(raw):
    _, y, g = raw
    for _, _, tr, te in protocol.make_splits(y, g):
        assert not set(g[tr]) & set(g[te])


def test_bootstrap_matches_concatenation():
    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(20), 3)
    y = np.tile([0, 1, 2], 20)
    p = np.where(rng.random(60) < 0.6, y, rng.integers(0, 3, 60))
    W = protocol.resample_weights(g, n_boot=5, seed=11)
    classes = np.unique(y)
    N = protocol._subject_confusions(y, g, p, classes)
    fast = protocol._ba_from_conf(np.einsum("bs,skl->bkl", W, N))
    from sklearn.metrics import balanced_accuracy_score
    for b in range(5):
        idx = np.concatenate([np.repeat(np.where(g == s)[0][None], int(w), 0).ravel()
                              for s, w in enumerate(W[b]) if w > 0])
        assert fast[b] == pytest.approx(balanced_accuracy_score(y[idx], p[idx]), abs=1e-12)


def test_within_subject_permutation_preserves_multisets(raw):
    _, y, g = raw
    yp = protocol.permute_within_subject(y, g, np.random.default_rng(0))
    for s in np.unique(g):
        assert sorted(yp[g == s]) == sorted(y[g == s])


def test_pooled_oof_and_permutation_end_to_end(legacy):
    y, g = legacy.y, legacy.groups
    F = features.erp_windows(legacy.X, legacy.times_ms).reshape(len(y), -1)
    splits = protocol.make_splits(y, g, seeds=(42,))
    preds = protocol.pooled_oof(lambda s, f, tr, te: {"clean": (F[tr], F[te], None)}, y, splits)
    ba = protocol.bootstrap_ba(y, g, preds, n_boot=200)["clean"]
    assert ba["ci95"][0] <= ba["BA"] <= ba["ci95"][1]
    cache = [(s, f, tr, te, F[tr], F[te]) for s, f, tr, te in splits]
    pt = protocol.permutation_test(cache, y, g, n_perm=9)
    assert pt["observed_BA"] == pytest.approx(ba["BA"])
    assert 0.1 <= pt["p"] <= 1.0


def test_pre_center_removes_offset_leak_keeps_scale(raw):
    X, y, g = raw
    X2 = X.copy()
    X2[:, 400:, :] += 3.0                                   # shift only post-stimulus samples
    pre = slice(0, C.ONSET - 10)
    for mode in ("pre_center", "epoch_z"):
        a = data.preprocess(X, y, g, "fir", mode).X
        b = data.preprocess(X2, y, g, "fir", mode).X
        if mode == "pre_center":                            # baseline mean stays exactly zero
            np.testing.assert_allclose(a[:, :C.ONSET].mean(axis=1), 0, atol=1e-9)
            np.testing.assert_allclose(b[:, :C.ONSET].mean(axis=1), 0, atol=1e-9)
        else:                                               # epoch_z moves the baseline offset
            assert np.abs(a[:, pre].mean(1) - b[:, pre].mean(1)).max() > 0.1


def test_lpp_window_is_mean_300_1000(legacy):
    f = features.lpp_window(legacy.X, legacy.times_ms)
    m = (legacy.times_ms >= 300) & (legacy.times_ms < 1000)
    np.testing.assert_allclose(f[:, :, 0], legacy.X[:, m, :].mean(axis=1))


def test_global_scale_is_one_scalar_and_count_independent():
    rng = np.random.default_rng(9)
    times = np.arange(-50, 250) * 4.0
    sig = np.where(times > 300, 5.0, 0.0)
    few = sig + rng.normal(0, 4.0, (50, 300)).mean(0) * 1.0      # noisy average (few trials)
    many = sig + rng.normal(0, 4.0, (800, 300)).mean(0)          # clean average (many trials)
    X = np.stack([few, many])[:, :, None]
    g = data.normalize(X, times, "global_scale")
    np.testing.assert_allclose(g * X.std(), X)                  # a single common scale
    b = data.normalize(X, times, "baseline_z")
    late = times > 300
    # baseline_z shrinks the noisier average's effect relative to the cleaner one
    assert b[0, late].mean() < 0.5 * b[1, late].mean()
    assert g[0, late].mean() == pytest.approx(g[1, late].mean(), rel=0.1)
