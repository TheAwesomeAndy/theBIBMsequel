# X06-2026-11: prevalence of epoch-own-statistics scaling (protocol)

Written and committed before any repository or paper below was inspected.

## Question

How often do widely used EEG decoding codebases and recent EEG decoding papers
scale each epoch, trial, or window by statistics computed from that same
epoch, trial, or window?

## Coding scheme (one code per item)

- **OWN**: scaling uses statistics of the same epoch/trial/window (per-epoch
  or per-window z-score, per-sample min-max, per-trial standardization,
  subtracting the epoch mean, dividing by the epoch's own SD or percentile).
  This class can leak post-stimulus information into earlier samples.
- **TRAIN**: statistics fitted on training data (or a fixed reference period
  outside the epoch, or the recording as a whole before epoching) and applied
  unchanged to test data.
- **PRE**: statistics from the pre-stimulus interval of the same epoch only.
- **FIXED**: a constant rescaling (unit conversion, fixed divisor).
- **NONE**: no amplitude scaling.
- **UNSTATED**: the code path or text does not determine the scaling.

An item that offers several options is coded by its default or by the option
used in its own reported experiments, and the alternatives are noted.

## Part A: codebases (fixed list)

braindecode, MOABB, TorchEEG, EEGNet reference code (ARL eegmodels), EEG
Conformer, BENDR, BIOT, LaBraM, EEGPT, CBraMod. For each, the default
preprocessing or the preprocessing of the repository's own fine-tuning scripts
is located and the deciding source line is recorded with its path.

## Part B: papers (fixed sampling rule)

arXiv API query `abs:EEG AND (abs:decoding OR abs:classification)` in
categories eess.SP, cs.LG, q-bio.NC, sorted by submission date descending,
starting 2025-12-31 and moving backward; the first 40 papers that report a
supervised EEG decoding experiment are included. Each paper's methods text is
coded with the scheme above, and the deciding sentence is quoted (at most one
short sentence). A second field records whether the decoded data are
stimulus- or event-locked epochs (where pre-event samples exist).

## Reporting

Counts per code, with every item, its code, and its evidence in
`outputs/X06/`. No claim beyond these counts is made.
