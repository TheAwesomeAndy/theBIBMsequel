# theBIBMsequel

Journal extension of the IEEE BIBM 2026 Doctoral Forum paper (S60204, accepted):

> A. A. Lane, W. Tang, and B. D. Nelson, "On the Edge of Stability: Spiking Reservoir
> State-Space Encoding of Affective EEG," IEEE BIBM 2026.

**Target journal:** IEEE Transactions on Neural Systems and Rehabilitation Engineering
(TNSRE), with IEEE JBHI as backup. Plan and status: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## What the journal paper adds

The conference paper showed that channel-dropout robustness rankings of affective-ERP
representations depend on the scoring protocol (zero-fill bias, the chance floor of raw
retention, augmentation state), using a fixed LIF reservoir as a near-centered reference.
The journal version:

1. fixes the reservoir window, which the BIBM camera-ready audit found was mostly
   pre-stimulus (-160 to +73 ms), and adds an anti-alias filter and a normalization that
   keeps post-stimulus signal out of the baseline;
2. adds a matched non-spiking control (Amp6: mean amplitude in the reservoir's own bins),
   so any reservoir contribution is measured against binning alone;
3. replaces the single reservoir weight draw with ten, and measures the threshold density
   the BIBM branching bound needed;
4. adds Proposition 3 (zero fill drives a non-centered encoder to chance under a change of
   origin) and a training-side fill-bias index (`docs/THEORY.md`);
5. adds spherical-spline and refit baselines, more datasets with montages, and more
   encoders (Phases 2 to 4).

## Layout

| Path | Contents |
|---|---|
| `sequel/` | implementation: data, reservoir, features, fills, protocol, theory |
| `experiments/` | one script per analysis; each writes one aggregate JSON |
| `tests/` | data-independent checks, incl. bit-exact agreement with the BIBM code |
| `outputs/aggregate/` | aggregate results only (none yet from real data) |
| `docs/` | roadmap and theory notes |
| `manuscript/` | journal manuscript skeleton (IEEEtran) and overlap log |
| `bibm_baseline/` | frozen BIBM camera-ready: source, analysis code, outputs, code audit |

## Running

```sh
pip install -r requirements.txt
python -m pytest -q tests                                  # 22 synthetic checks

# Gate: reproduce the BIBM numbers with the new code (needs the BIBM pickle)
python experiments/check_bibm_regression.py --pickle /path/shape_features_211.pkl

# Phase 1 on SHAPE (needs the three-condition directory)
python experiments/run_window_study.py      --shape-dir /path/batch_data
python experiments/run_reservoir_dynamics.py --shape-dir /path/batch_data
python experiments/run_permutation.py        --shape-dir /path/batch_data --n-perm 1000 --jobs 4
```

Add `--synthetic --quick` to any experiment for a code check on the synthetic cohort
(output files end in `_SYNTHETIC.json` and carry no scientific meaning).

## What the tests do and do not show

They show that the code reproduces the BIBM reservoir and band power exactly, that the
preprocessing boundaries hold (anti-aliasing, no post-to-pre leakage under `baseline_z`),
that Propositions 1 and 3 hold exactly for the implemented readout, and that the
bootstrap and permutation machinery behave as specified. They do not show that any
empirical result is true.

## Data

SHAPE (Laboratory for Clinical Affective Neuroscience, Stony Brook University) is
restricted and never stored here. DEAP is available from its authors under its licence.
