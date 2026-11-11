# Publication-overlap log

## Submission

**Target:** IEEE BIBM 2026  
**Working title:** On the Edge of Stability: Spiking Reservoir State-Space Encoding of Affective EEG

## Purpose

This log records the boundary between the BIBM conference paper and the later journal manuscript. The conference paper is a narrow technical paper on reservoir temporal encoding. It should not become a shortened version of the full operational-distinctness journal paper.

## Included in BIBM conference paper

- ERP observation framing as noisy biomedical time-series measurement.
- Fixed LIF reservoir encoding.
- Six-bin spike-count encoding, BSC6.
- PCA-compressed reservoir embedding E.
- Subject-level validation.
- Classical ERP/bandpower baseline comparison.
- Representative perturbation diagnostics for amplitude noise, temporal jitter, and channel dropout.
- Bounded interpretation: reservoir as temporal measurement operator, not universal classifier.

## Reserved for journal paper

- Full observable map Phi = [E, D, T, C].
- Full operational-distinctness framework.
- Graph-topological tPLV stream T.
- Structure-function coupling kappa stream C.
- Full CKA/redundancy analysis.
- Full perturbation-regime matrix across all observable streams.
- Closed-loop expected-free-energy/evidence-accumulation experiments.
- Full TAFFC affective-computing positioning.
- Any broad diagnostic-biomarker claims.

## Manuscript-language guardrails

Use:

- measures
- characterizes
- estimates
- preserves
- transforms
- suppresses
- operating regime
- subject-level validation

Avoid:

- state-of-the-art
- breakthrough
- diagnostic biomarker
- low-power wearable feasibility
- clinical deployment
- ARSPI-Net as title lead
- hybrid model as novelty

## Numerical provenance

All numerical anchors in the manuscript were verified against source tables/outputs in
`TheAwesomeAndy/operational-distinctness-paper`:

- Dataset (211 subjects after excluding subject 127, 633 observations, 3 conditions, 34 channels,
  256 Hz, 256 post-stimulus samples): `outputs/tcds_ready9/dataset_provenance_metadata.json`,
  `tables/tcds_ready9/table_dataset_provenance.tex`.
- Clean performance (A0 0.485/0.483, A1 0.463/0.460, A2 0.432/0.429, A6 0.478/0.475;
  E dim 2176): `tables/tcds_ready9/table_mechanism_ablation.tex`.
- Representation-level perturbation (A0/A1 under amplitude noise and channel dropout):
  `tables/tcds_ready9/table_robustness_summary.tex`,
  `outputs/tcds_ready9/analysis/robustness_summary.csv`.
- Raw-signal recomputation (E recomputed): `outputs/tcds_ready9/analysis/robustness_summary.csv`.

Correction reported: the macro-F1 column of the raw-signal recomputation table was brought into
agreement with `robustness_summary.csv` (clean 0.426, jitter 50 ms 0.352, amplitude 5 dB 0.419,
dropout 30% 0.233) and the clean A2 macro-F1 was set to 0.429; the prior draft values did not match
the source outputs. No new experiments were run and no figures were regenerated.

## CV listing if accepted

List as an IEEE BIBM proceedings paper unless official IEEE metadata explicitly labels the paper differently.

Suggested citation format:

Andrew A. Lane, W. Tang, and B. D. Nelson, "On the Edge of Stability: Spiking Reservoir State-Space Encoding of Affective EEG," in Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Dallas, TX, USA, 2026.
