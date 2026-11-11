# Overlap with the BIBM 2026 paper

The cover letter must disclose the conference paper and list what is new. Keep this file
current; it is the source for that list.

## Reused from BIBM (to be cited, not presented as new)

- Propositions 1 and 2 (fill rule and re-centering; chance floor of raw retention)
- SHAPE three-condition cohort, subject-grouped protocol, subject-level bootstrap
- Reservoir definition (256 LIF units, beta 0.05, theta 0.5, BSC6)
- DEAP band-power fill check (if kept, extended)

## New in the journal version (fill in as results arrive)

- [ ] Corrected preprocessing: FIR anti-alias decimation, baseline-only normalization,
      post-stimulus reservoir windows
- [ ] Matched non-spiking control (Amp6) for the reservoir
- [ ] Ten reservoir weight draws; measured threshold density and branching threshold
- [ ] 1000-permutation null with the headline estimator
- [ ] Proposition 3 and the fill-bias index, validated across encoders and datasets
- [ ] Spherical-spline and refit baselines
- [ ] Additional datasets and encoders (Phases 2 and 3)

## Changes to BIBM numbers

Record here any BIBM value that changes under the corrected preprocessing, with the
reason, so the paper can state it explicitly.
