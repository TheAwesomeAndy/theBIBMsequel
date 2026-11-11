# Journal extension roadmap

**Target:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE).
Backup: IEEE Journal of Biomedical and Health Informatics (JBHI).
Before submission, check both journals' current author guidelines on extending a
conference paper (amount of new material, disclosure in the cover letter) and whether the
BIBM 2026 Doctoral Forum paper appears in IEEE Xplore.

**Working framing.** The BIBM paper's results are about how channel-dropout robustness is
scored (fill rule, chance floor, augmentation), with the reservoir as a near-centered
reference. The journal paper makes that the headline and generalizes it across datasets
and encoders. Title is the author's decision (BIBM memory #23 kept "On the Edge of
Stability" for the conference version).

Every phase ends with a report and an explicit go-ahead before the next one starts.

## Phase 0: repository and code (done in this branch)

- [x] Frozen BIBM camera-ready under `bibm_baseline/` (source commit 1667277)
- [x] `sequel/` package: corrected preprocessing, reservoir with membrane statistics,
      matched Amp6 control, fills incl. spherical spline and refit, fast subject bootstrap,
      permutation test, Proposition-1 diagnostics
- [x] 22 synthetic tests, incl. bit-exact agreement with the BIBM reservoir and band power
- [x] Phase-1 experiment scripts, smoke-tested on synthetic data

## Phase 1: run on SHAPE (needs the restricted data; author runs locally)

1. `python experiments/check_bibm_regression.py --pickle <shape_features_211.pkl>`
   Gate: all 12 BIBM cells reproduce to 1e-6. Nothing else is trusted until this passes.
2. `python experiments/run_window_study.py --shape-dir <batch_data>` (about 2 to 4 h)
3. `python experiments/run_reservoir_dynamics.py --shape-dir <batch_data>` (about 2 to 3 h)
4. `python experiments/run_permutation.py --shape-dir <batch_data> --n-perm 1000 --jobs 4`

Decision gate after Phase 1:
- Which preprocessing and window become primary (expected: FIR decimation, baseline_z,
  post-stimulus window). The BIBM window was mostly pre-stimulus (audit A18).
- Does BSC6 beat the matched Amp6 control? If not, the reservoir stays only as the
  near-centered reference and the paper says so.
- Reconcile with the dissertation's confirmatory pipeline (64.3%, different window and
  per-channel reservoirs) in one sentence of the paper.

## Phase 2: datasets

- [ ] SHAPE electrode montage from the lab (needed for spherical-spline fill and any
      spatial statement)
- [ ] Public ERP datasets with montages: ERP CORE; one P300 BCI set (core TNSRE audience)
- [ ] TCRZEM (IAPS passive viewing, loader exists in the dissertation repo)
- [ ] DEAP extended beyond band power

## Phase 3: encoders

- [ ] EEGNet with and without augmentation (port BIBM `experiment_eegnet_signal.py`, needs torch)
- [ ] Riemannian tangent space (pyriemann)
- [ ] Removed versus zero-padded channels (engages Sirca et al. 2026 directly)
- [ ] Linear probe on one EEG foundation model

## Phase 4: theory validation

- [ ] Fill-bias index versus observed zero-fill loss across encoders, rates, datasets
- [ ] Re-centering experiment: shift each encoder's origin and confirm Proposition 3
- [ ] Multiple electrode-mask draws per fold (BIBM used one)

## Phase 5: manuscript

- [ ] `manuscript/main.tex` (IEEEtran journal), 40 to 60 references, every entry checked
      against Crossref or arXiv
- [ ] Update `manuscript/overlap_log.md` with what is new relative to BIBM
- [ ] Author wording rules (`bibm_baseline/memory_bibm.jsonl` #16) enforced by script
