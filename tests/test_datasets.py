"""Checks for the Phase 2 dataset layer (no downloads; synthetic inputs only)."""
from __future__ import annotations

import numpy as np
import pytest

from sequel import data, datasets


def test_cut_epochs_matches_shape_layout():
    sf = 1024.0
    cont = np.tile(np.arange(20000.0)[:, None], (1, 3))
    ep, n_pre, ok = datasets.cut_epochs(cont, sf, np.array([1.0, 5.0, 19.9]))
    assert n_pre == 205 and ep.shape[1] == 1229            # SHAPE: onset 205 of 1229
    assert ok.tolist() == [True, True, False]               # last epoch leaves the recording
    np.testing.assert_allclose(ep[:, :n_pre].mean(axis=1), 0.0, atol=1e-9)   # baseline-corrected


def test_average_conditions_rejects_and_requires_trials():
    rng = np.random.default_rng(0)
    tr = rng.normal(0, 1, (40, 50, 4))
    tr[3, 10, 0] = 1e3                                       # one artifact trial
    lab = np.repeat([0, 1], 20)
    avg, counts = datasets.average_conditions(tr, lab, [0, 1], reject_uv=200, min_trials=10)
    assert avg.shape == (2, 50, 4) and counts == [19, 20]
    none, _ = datasets.average_conditions(tr, lab, [0, 1], reject_uv=200, min_trials=25)
    assert none is None


@pytest.mark.parametrize("fs,onset", [(1024, 205), (512, 102), (256, 51)])
def test_generic_decimation_lands_on_onset(fs, onset):
    T = onset + fs                                           # -onset .. +1 s
    raw = np.random.default_rng(1).normal(size=(2, T, 3))
    X, times = data.decimate_raw(raw, "fir", fs_raw=fs, onset_raw=onset)
    k = int(np.argmin(np.abs(times)))
    assert abs(times[k]) < 1e-9
    assert np.allclose(np.diff(times), 1000.0 / 256)


def test_legacy_decimation_guarded():
    with pytest.raises(ValueError):
        data.decimate_raw(np.zeros((1, 614, 2)), "legacy", fs_raw=512, onset_raw=102)


def test_erpset_roundtrip_and_preprocess(tmp_path):
    rng = np.random.default_rng(2)
    s = datasets.ERPSet("toy", rng.normal(size=(6, 614, 5)), np.tile([0, 1, 2], 2),
                        np.repeat(["a", "b"], 3), 512, 102, list("ABCDE"), ["x", "y", "z"],
                        rng.normal(size=(5, 3)), np.arange(6), {"k": 1})
    s.save(tmp_path / "toy.npz")
    t = datasets.ERPSet.load(tmp_path / "toy.npz")
    np.testing.assert_array_equal(t.X, s.X)
    assert t.ch_names == s.ch_names and t.fs_raw == 512 and t.info == {"k": 1}
    ep = t.to_epochs()
    assert abs(ep.times_ms[ep.onset]) < 1e-9 and ep.X.shape[2] == 5


def test_vhdr_parser(tmp_path):
    h = tmp_path / "S1_Mastoid_Neg_High.vhdr"
    h.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
                 "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\n"
                 "SamplingInterval=1953.125\nSegmentDataPoints=1792\n[Binary Infos]\n"
                 "BinaryFormat=INT_16\n[Channel Infos]\nCh1=Fz,,0.1,µV\nCh2=VEOG,,0.5,µV\n")
    info = datasets._vhdr(h)
    assert info["names"] == ["Fz", "VEOG"] and info["sfreq"] == pytest.approx(512.0)
    np.testing.assert_allclose(info["res"], [0.1, 0.5])
    assert info["segment_points"] == 1792


def test_template_positions_known_and_unknown():
    pytest.importorskip("mne")
    assert datasets.template_positions(["Fz", "Cz", "Pz"]).shape == (3, 3)
    assert datasets.template_positions(["Fz", "NotAChannel"]) is None


def test_regress_eog_removes_ocular_component():
    rng = np.random.default_rng(5)
    n = 20000
    blink = np.zeros(n)
    blink[rng.integers(0, n - 200, 40)[:, None] + np.arange(200)] = 150.0  # 150 uV blinks
    brain = rng.normal(0, 5, (n, 4))
    eeg = brain + np.outer(blink, [1.0, 0.6, 0.2, 0.05])
    clean = datasets.regress_eog(eeg, (blink + rng.normal(0, 1, n))[:, None])
    # A constant offset (the mean of the removed component) remains by design; epochs are
    # baseline-corrected afterwards, so compare mean-removed signals.
    err = (clean - clean.mean(0)) - (brain - brain.mean(0))
    raw_err = (eeg - eeg.mean(0)) - (brain - brain.mean(0))
    assert np.abs(err).max() < 0.05 * np.abs(raw_err).max()
