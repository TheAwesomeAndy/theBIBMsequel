# Paper analysis code (BIBM 2026 Doctoral Forum, paper S60204)

This folder holds the code that produces every number, table, and figure in
`manuscript/main_bibm2026.tex`. It is separate from the research harness
(`scripts/`, `papercheck/`, `harness/`, `tools/`, `makefiles/`, `playbooks/`,
`.claude/`), which is synced from the audit repository (see the top-level README).

All scripts read restricted data **locally** and write only aggregate,
de-identified JSON to `outputs/aggregate/` (or a figure PDF). Nothing
subject-level is ever written to the repository.

## Data

| Input | Where it comes from | Environment variable |
|---|---|---|
| SHAPE feature pickle (`shape_features_211.pkl`: `X_ds`, `y`, `subjects`) | restricted; held outside this repository | `ARSPI_SHAPE_FEATURES` (default: sibling `operational-distinctness-paper/data/`) |
| DEAP `data_preprocessed_python` (`s01.dat`…`s32.dat`) | DEAP authors, under the DEAP end-user licence | `DEAP_DIR` (default `/home/user/data_local/deap`) |

The pickle's `conv_feats` field is **not used**: it cannot be reproduced from
`X_ds` with the documented band-power estimator (see `docs/CODE_AUDIT.md`), so
band power is always recomputed from the analysed signal.

## Common protocol

StratifiedGroupKFold (5 folds) × seeds 42–46 with subjects as groups;
train-only standardization; balanced L2 logistic readout; out-of-fold class
probabilities averaged per observation over the five partitions; 95% CIs from a
subject-level bootstrap (211 subjects resampled with replacement, each keeping
its three condition observations; 4000 resamples); encoder or fill contrasts as
paired differences under the same resampling. The reservoir PCA basis is fitted
on the training subjects of each fold unless a script says otherwise.

## Map: paper element → script → output

| Paper element | Script | Output |
|---|---|---|
| Fig. 1 (reservoir dynamics; accuracy vs ρ; measured transition) | `make_fig_reservoir_dynamics.py` (needs `e1_rho_sweep.json`) | `manuscript/figures/imported/fig_reservoir_dyn.pdf` |
| Spectral-radius sweep (ρ = 0…1.5, ρ = 0 ablation, damage spreading) | `experiment_camera_ready.py e1` | `e1_rho_sweep.json` |
| Fig. 2 (grand-average affective ERP) | `make_fig_erp.py` | `manuscript/figures/imported/fig_erp.pdf` |
| Table I (clean BA [CI], macro-F1, macro OvR AUC); permutation null; PCA component count; BSC₁ vs BSC₆; encoder centeredness | `experiment_controls.py` | `e5_controls.json` |
| Table II, Fig. 3 (feature-coordinate fill experiment, train-only PCA; Fig. 3 plots the paired fill-minus-zero contrasts at every dropout level, `fill_minus_zero`) | `experiment_camera_ready.py e2`; figure: `make_fig_subject_level.py` (also reads `e5_controls.json` for the centeredness legend) | `e2_trainonly_pca_fills.json`; `fig_impute_subj.pdf` |
| Table III (signal level: electrode removal 10/30/50%, 5 dB noise, ±50 ms jitter; all encoders; EEGNet paired augmented-minus-unaugmented and clean-minus-condition contrasts) | `experiment_signal_perturbations.py` (band-power, ERP-window, reservoir), `experiment_eegnet_signal.py` (EEGNet ± augmentation) | `e4_signal_perturbations.json`, `e3_eegnet_signal.json` |
| Fig. 4 (DEAP fill-rule check; EEGNet training effect) | `deap_replication.py`; figure: `make_fig_deap_aug.py` | `deap_replication_v2.json`; `fig_deap_aug.pdf` |
| Transductive (pooled-PCA) sensitivity | `regen_reservoir_embedding.py` → `reanalysis_subject_bootstrap.py` | `subject_bootstrap_reanalysis.json` |
| Shared helpers (band power, ERP windows, fills, subject bootstrap) | `experiment2_rawsignal.py`, `reanalysis_subject_bootstrap.py` | — |
| Shared signal-level perturbations (identical draws for every encoder) | `perturbations.py` | — |

## Run order

```sh
python analysis/experiment_camera_ready.py e1        # ~30 min on 4 cores
python analysis/experiment_camera_ready.py e2        # ~20 min
python analysis/experiment_signal_perturbations.py   # ~30 min
python analysis/experiment_eegnet_signal.py          # ~75 min (CPU, PyTorch)
python analysis/experiment_controls.py               # ~20 min
python analysis/deap_replication.py                  # ~10 min (needs DEAP_DIR)
python analysis/make_fig_reservoir_dynamics.py
python analysis/make_fig_erp.py
python analysis/make_fig_subject_level.py            # after E2 and E5
python analysis/make_fig_deap_aug.py                 # after E3 and the DEAP run
python analysis/export_paper_values.py               # paper-values CSV for the table gate
```

Then build the paper (`manuscript/BUILD.md`) and run the value gate
(`python scripts/check_table_manifest.py harness/table_manifest.yaml`).

## Superseded outputs

`outputs/aggregate/controls_*.json`, `eegnet.json`, `eegnet_aug.json`, and
`deap_replication.json` were produced before the camera-ready by scripts that
were never committed. They are kept for provenance only; no number in the
camera-ready manuscript depends on them. `experiment2_rawsignal.json` (the
pre-camera-ready signal-level table) is superseded by `e4_signal_perturbations.json`, which embeds
training and test reservoir codes with the same PCA transform (see `docs/CODE_AUDIT.md`, A10).


## Journal extension (theBIBMsequel)

The journal experiments run on SHAPE (`JDATA=shape`) and on the public external cohort (DataVerseNL
doi:10.34894/TCRZEM; set `TCRZEM_DIR` to the export directory, default
`/home/user/data_local/tcrzem/export`; single trials are cached under `TCRZEM_CACHE`).
`TCRZEM_EPOCH=long` selects the pre-specified long epoch (-200..+2500 ms). All scripts
write aggregate JSON to `outputs/aggregate/journal/`.

```sh
python analysis/tcrzem_data.py                         # build the trial cache, print the cohort
for E in standard long; do
  TCRZEM_EPOCH=$E python analysis/j1_core_replication.py   # ~20 min
  TCRZEM_EPOCH=$E python analysis/j2_signal_level.py       # 15 min (standard) to 2 h (long)
  TCRZEM_EPOCH=$E python analysis/j5_origin_shift.py
  TCRZEM_EPOCH=$E python analysis/j6_prestimulus.py
  TCRZEM_EPOCH=$E python analysis/j7_prestim_null.py
done
python analysis/j3_eegnet.py                                         # all seeds, ~2 h CPU
TCRZEM_EPOCH=long J3_SEEDS=42 python analysis/j3_eegnet.py           # seed-42 partition
for R in default epochs40 epochs160 dropout50; do J3_RECIPE=$R J3_SEEDS=42 python analysis/j3_eegnet.py; done
TCRZEM_EPOCH=long python analysis/j4_edge_of_stability.py            # ~2.5 h
TCRZEM_EPOCH=long python analysis/j8_silence_distance.py             # ~25 min
JDATA=tcrzem python analysis/j9_zscore_hypothesis.py                 # ~2 min
# SHAPE (restricted, local only; see CLAUDE.md): verify the input, then the same analyses
python analysis/shape_data.py                    # X_ds == zscore(raw[0:1024:4]) for all 633 observations
export JDATA=shape
for W in standard onset post; do                 # conference windows / windows from true onset
  SHAPE_EPOCH=$W python analysis/j1_core_replication.py     # ~25 min
  SHAPE_EPOCH=$W python analysis/j4_edge_of_stability.py    # ten draws, ~75 min
  SHAPE_EPOCH=$W python analysis/j5_origin_shift.py
  SHAPE_EPOCH=$W python analysis/j8_silence_distance.py
done
python analysis/j6_prestimulus.py; python analysis/j7_prestim_null.py; python analysis/j9_zscore_hypothesis.py
unset JDATA
python analysis/make_fig_tcrzem_erp.py; python analysis/make_fig_fills_two_cohorts.py
python analysis/make_fig_origin.py; python analysis/make_fig_prestimulus.py
python analysis/make_fig_edge.py; FIG_EPOCH=_long python analysis/make_fig_signal.py
python analysis/export_journal_values.py
```

| Journal element (numbers from the compiled PDF) | Script | Output |
|---|---|---|
| Fig. 1 (reservoir dynamics, SHAPE) and Fig. 2 (SHAPE ERPs) | conference scripts, unchanged | `manuscript/figures/imported/` |
| Fig. 3 (external ERPs) | `make_fig_tcrzem_erp.py` | `manuscript/figures/journal/fig_tcrzem_erp.pdf`, `tcrzem_erp_summary.json` |
| Table I (clean, both cohorts) | J1, J3 (+ conference E3, E5) | `manuscript/journal/generated/tab_clean.tex` |
| Table II (SHAPE conference vs onset vs post windows) | J1, J4 on SHAPE (`SHAPE_EPOCH`) | `generated/tab_windows.tex` |
| Table III, Fig. 4 (fills at 30%, fill contrasts) | J1 (+ conference E2); `make_fig_fills_two_cohorts.py` | `generated/tab_fills.tex`, `fig_fills.pdf` |
| Fig. 5, Table IV (origin test; SHAPE table) | J5 on both cohorts; `make_fig_origin.py` | `fig_origin.pdf`, `generated/tab_origin_sj.tex` |
| Fig. 6 (pre-stimulus windows) and the z-score test | J6, J7, J9; `make_fig_prestimulus.py` | `fig_prestimulus.pdf` |
| Fig. 7, Table V (edge of stability; SHAPE ten draws) | J4, J8 on both cohorts; `make_fig_edge.py` | `fig_edge.pdf`, `generated/tab_edge_s.tex` |
| Fig. 8 (DEAP, SHAPE EEGNet) | conference script, unchanged | `manuscript/figures/imported/fig_deap_aug.pdf` |
| Table VI (SHAPE signal level) | conference E3, E4 | `generated/tab_signal_shape.tex` |
| Table VII, Fig. 9 (external signal level, long epoch) | J2, J3; `make_fig_signal.py` | `generated/tab_signal_xl.tex`, `fig_signal_long.pdf` |
| Tables VIII-X (appendix: matched-epoch signal level, external origin test, external edge) | J2, J3, J4, J5 | `generated/tab_signal_x.tex`, `tab_origin_xl.tex`, `tab_origin_x.tex`, `tab_edge.tex` |
| Table XI (appendix: EEGNet recipes) | J3 recipe runs | `generated/tab_recipes.tex` |
| Every inline number | `export_journal_values.py` | `generated/values.tex`, `outputs/aggregate/journal/journal_values.csv` |

`python scripts/check_journal.py` regenerates the values, builds the PDF, and runs the
journal gates; `python analysis/verify_jcore.py` checks the port against the conference code.
