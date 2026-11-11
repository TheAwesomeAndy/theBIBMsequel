# SHAPE data: provenance and format

Source: the data provider's export notes for the SHAPE Community emotional interrupt task
(sent at the original data handoff, about four years before this extension), plus the
checks in this repository. For the methods section.

## Export format (from the provider)

- Three files per participant: neutral, pleasant, unpleasant (condition in the file name,
  `SHAPE_Community_{ID}_IAPS{Neu,Pos,Neg}_BC.txt`).
- Sampling rate 1024 Hz. Segment 1200 ms: 200 ms pre-stimulus baseline and 1000 ms after
  onset, so 1229 rows per file.
- 34 electrodes, one per column. The header naming the electrodes was removed before
  export; the provider can supply the column-to-electrode mapping on request.
- The provider recommended an automated check that every file is 1229 x 34.
- The provider recommended removing the first 205 rows (the 200 ms baseline) before
  analysis.

## Related published methods (same lab, different cohort)

Nelson et al., 2015, Psychol. Med. 45:2545-2556 (PMC4702485): modified emotional interrupt
task (40 neutral, 40 pleasant, 40 unpleasant IAPS trials; picture 1000 ms), BioSemi
ActiveTwo, 34 electrodes on the 10/20 system plus mastoids, mastoid reference, 0.1-30 Hz,
Gratton ocular correction, LPP = mean 300-1000 ms at Pz/P3/P4 and Oz/O1/O2. Whether every
step applies to the SHAPE Community export is to be confirmed with the lab.

## What this repository verified

- `experiments/check_shape_files.py` -> `outputs/aggregate/shape_file_check.json`:
  636 files, 212 participants, every file 1229 x 34, complete panels, no non-finite
  values, no duplicate files, zero-mean baselines. Only anomaly: participant 127 (Neu flat
  on all 34 channels, Neg flat on 6, Pos flat on 2; largest value in the export, 326 uV).
  127 is excluded, leaving 211 participants and 633 observations (as in BIBM).
- Spectrum falls about 47 dB from 10 to 100 Hz, consistent with a 30 Hz low-pass upstream.
- `experiments/check_shape_montage.py`: channels 0-31 match the BioSemi-32 standard order
  (Mantel r = -0.673 vs a random-order null of 0.000 +- 0.046); 32 and 33 behave like FCz
  and Iz. Inference only; the provider's mapping should replace it.

## How the analyses use the baseline

The BIBM reservoir window (-160..+74 ms) lay mostly inside the baseline the provider
recommends removing; under baseline-only normalization its accuracy is at chance (0.305).
The journal analyses take features from post-stimulus samples only. The baseline is used,
if at all, only for normalization statistics (`baseline_z`, `pre_center`), and the
recommended primary normalization (`global_scale`) does not use it.
