# Relation to the conference version

**Conference paper.** A. A. Lane, W. Tang, and B. D. Nelson, "On the Edge of
Stability: Spiking Reservoir State-Space Encoding of Affective EEG," IEEE BIBM 2026
(Doctoral Forum, paper S60204), accepted.

**Journal manuscript.** "Encoding Nothing at the Edge of Stability: Spiking Reservoir
Representations of Affective EEG" (`main_journal.tex`).

## What is reused

- The SHAPE results (Tables I-III and Figs. 1-4 of the conference paper) are reported
  as conference results and labeled as such in every table and caption. No SHAPE or
  DEAP analysis was rerun for the journal version.
- Proposition 1 and the observation model are carried over, with the proof unchanged.
- Parts of the Methods text (reservoir, BSC6, baselines, validation) are adapted from
  the conference paper.

## What is new

1. **External cohort.** An independent public IAPS cohort (DataVerseNL
   doi:10.34894/TCRZEM, 228 subjects, different laboratory and montage), analyzed at the
   conference-matched epoch and at a long epoch pre-specified from the dataset's published
   effect latencies: clean metrics, permutation null, PCA controls, BSC1 vs BSC6,
   centeredness, the four-fill dropout experiment, and signal-level perturbations.
2. **Direct test of Proposition 1.** An origin sweep that moves the fill along the
   standardized axis: clean predictions do not change, dropout accuracy follows the
   fill position, and the zero-versus-mean choice rewrites predictions in proportion to
   feature non-centeredness.
3. **Proposition 2 and the pre-stimulus experiments.** Per-epoch z-scoring writes the
   post-stimulus mean into the pre-stimulus baseline; pre-stimulus-only features decode
   affect above chance under that normalization and not under a fixed scale (with a
   200-permutation null); a post-onset reservoir window is compared with the conference
   window, which lies mostly before onset.
4. **Edge of stability across draws.** The damage-spreading transition for ten reservoir
   draws, three thresholds, and four leaks, and accuracy versus spectral radius for five
   draws.
5. **Physically grounded perturbations.** Spherical-spline repair of removed electrodes
   and re-averaging test observations from 1, 4, or 16 recorded trials.
6. **Trained baseline.** EEGNet with and without augmentation on the second cohort, and
   recipe variation (epochs, dropout).
7. **Reporting standard.** Five reporting items derived from the two-cohort results.

All new results come from committed scripts (`analysis/j*.py`) that write aggregate
JSON only, and every number in the manuscript is generated from that JSON
(`analysis/export_journal_values.py`).
