# Relation to the conference version

**Conference paper.** A. A. Lane, W. Tang, and B. D. Nelson, "On the Edge of
Stability: Spiking Reservoir State-Space Encoding of Affective EEG," IEEE BIBM 2026
(Doctoral Forum, paper S60204), accepted.

**Journal manuscript.** "The Weight of Nothing: Missing Electrodes, Empty Baselines, and
the Edge of Stability in Spiking Reservoir Encoding of Affective EEG" (`main_journal.tex`).

## What is reused

- Proposition 1 and the observation model are carried over, with the proof unchanged.
- Parts of the Methods text (reservoir, BSC6, baselines, validation) are adapted from
  the conference paper.
- Figs. 1, 2 and 8 (reservoir dynamics, SHAPE ERPs, DEAP and SHAPE EEGNet) and the SHAPE
  signal-level and EEGNet tables are the conference results, labeled as such. The DEAP
  check was not rerun.

## What is new

1. **SHAPE rerun.** Every new experiment that subject averages and an undocumented
   montage allow also runs on SHAPE (not spline repair, trial-count noise, or EEGNet).
   The journal code reproduces the conference fixed-encoder clean results, fills, and
   draw-42 operating-point sweep exactly; the pooled-basis sensitivity value differs by
   one observation of 633 (an earlier regeneration of the embedding). Analysis windows
   measured from true onset (the window the submitted conference manuscript described
   before its camera-ready correction, +39..+273 ms, and the whole post-onset interval,
   0..+797 ms) were fixed before their results were computed.
2. **External cohort.** An independent public IAPS cohort (DataVerseNL
   doi:10.34894/TCRZEM, 228 subjects, different laboratory and montage), analyzed at the
   conference-matched epoch and at a long epoch pre-specified from the dataset's published
   effect latencies.
3. **Direct test of Proposition 1** on both cohorts: an origin sweep that moves the fill
   along the standardized axis. Clean predictions do not change; the dropout score falls
   as the fill moves away from the training mean.
4. **Proposition 2 and the pre-stimulus experiments.** Per-epoch z-scoring writes the
   post-stimulus mean into the pre-stimulus baseline. On SHAPE, pre-stimulus features
   classify affect far above chance under that normalization and at chance under a fixed
   scale, and the conference reservoir window read the normalization offset (its code
   predicts the post-stimulus mean with R^2 = 0.999) rather than evoked dynamics. Placed
   after onset, the reservoir measures the evoked signal. This changes the interpretation
   of the conference reservoir result and is stated in the paper.
5. **Edge of stability across draws.** The damage-spreading transition and accuracy for
   ten reservoir draws on SHAPE (with a decision rule fixed in advance for the recurrence
   claim) and on the external cohort, and the transition across thresholds and leaks.
6. **Physically grounded perturbations** (external cohort): spherical-spline repair of
   removed electrodes and re-averaging test observations from 1, 4, or 16 recorded trials.
7. **Trained baseline.** EEGNet with and without augmentation on the second cohort, and
   recipe variation (epochs, dropout).
8. **Reporting standard.** Reporting items derived from the two-cohort results.

All new results come from committed scripts (`analysis/j*.py`) that write aggregate
JSON only, and every number in the manuscript is generated from that JSON
(`analysis/export_journal_values.py`).
