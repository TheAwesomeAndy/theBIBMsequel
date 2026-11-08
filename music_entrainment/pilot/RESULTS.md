# NMED-T week-1 feasibility pilot: results (2026-11-08)

Scope: go/no-go checks only. Nothing here is a confirmatory result. Three cleaned
NMED-T song files (songs 21, 28, 30; 56, 120, 150 BPM; 20 participants each,
125 Hz, 125 rows) were downloaded from the Stanford Digital Repository
(CC BY). No audio was used (none is distributed).

Reservoir: the canonical `chapter5Experiments/experiment_zero.LIFReservoir`
(256 LIF, beta 0.05, theta 0.5, rho 0.9, seed 42), driven by the z-scored
group-PC1 spatial component (Losorelli et al. 2017 filter) resampled 125 -> 256 Hz,
and by each participant's own projection on that filter (single listen).

## 1. Data access and integrity

| Check | Result |
|---|---|
| File manifest | 55 files, 39.7 GB; cleaned `songSS_Imputed.mat` 656-722 MB each (MATLAB v7.3 / HDF5) |
| Audio | not distributed; Table 1 gives Amazon ASINs only |
| Nominal tempi vs tapping | all 10 songs within ~1 % at the median after octave correction (mean ITI; tap times are quantized to 4096/44100 s = 92.9 ms, 98.8 % of taps on that grid) |
| Nominal tempi vs EEG | song 28: refined 120.006 BPM (+0.005 %); song 30: 150.018 BPM (+0.012 %); song 21: 56.26 BPM (+0.5 %, weak peak z 2.7, unreliable) |
| Load | 125 x 36513 x 20 (song 28), no NaN after imputation |

## 2. Reservoir under continuous drive

282 s of drive (72,000 steps at 256 Hz) runs in 0.9 s wall time; mean firing rate
0.126 on every drive tested (SHAPE operating point was 0.12); no silent units.
The continuous regime is feasible and the firing rate matches the characterized one.

## 3. Beat tagging: drive vs reservoir readouts (neighbour-frequency z, Welch 40 s segments)

Group-PC1 drive, z at (1, 2, 4, 8) x tempo:

| Song | tempo | drive | reservoir population rate | reservoir unit-PC1 (unsupervised linear readout over 256 unit rates) | units with z > 1.96 (chance 0.025) |
|---|---|---|---|---|---|
| 21 | 0.93 Hz | -0.6, 0.2, 0.8, -0.2 | 1.3, 1.7, -1.7, -0.3 | not run (no drive tagging) | - |
| 28 | 2.00 Hz | 1.1, 8.9, **24.7**, -0.4 | -1.0, -1.4, 1.9, 2.9 | 0.9, 9.3, **22.2**, -0.1 | 0.06, 0.56, **0.77**, 0.05 |
| 30 | 2.50 Hz | **12.6, 21.4**, 7.3, 0.9 | 0.1, -0.5, -1.2, 2.5 | **12.5, 18.2**, 6.4, 1.3 | 0.66, 0.64, 0.49, 0.08 |

Single-listen detection (fraction of 20 participants with z > 1.96 at the best harmonic):
song 28, 8 Hz: drive 0.70 (median z 3.4), population rate 0.05 (median z 0.3);
song 30, 5 Hz: drive 0.50 (median z 1.9), population rate 0.15.

Readings:
1. Beat-related tagging is present and strong in these data at the group level
   for songs 28 and 30, mostly at 2x and 4x the tempo (metrical subdivisions,
   consistent with Losorelli et al. 2017 Fig. 2), and absent at the group level
   for song 21. Tagging strength is song-dependent, not just tempo-dependent.
2. The reservoir population mean rate destroys the tagging (units lock at
   different phases and cancel). Per-unit rates retain it; an unsupervised
   linear readout across units recovers ~90 % of the drive's z. The proposed
   population-rate MI is the wrong readout; a per-neuron beat-cycle readout is
   the minimum.
3. In the canonical configuration the reservoir does not add detectability
   over its own linear input (unit-PC1 z <= drive z on both songs). Any added
   value must come from nonlinear structure the linear statistic cannot see,
   which is exactly what the evoked-twin design has to isolate.

## 4. Null models: what the pilot showed is broken

| Null in the proposal | What happened | Why |
|---|---|---|
| Circular shift of EEG vs beat grid, MI or locking magnitude | observed statistic inside the null on every drive, including the group drive with z 24.7 spectral tagging | for a strictly periodic grid a shift only moves the phase origin; MI and \|sum r e^{-ik phi}\| are invariant to it |
| Phase-randomized drive ("evoked twin" residual surrogate) through the reservoir | per-unit tagging identical to real drive (song 28: 82 % vs 77 % of units; song 30: 65 % vs 66 %) | phase randomization preserves the spectral line at k x tempo, so the surrogate is just as beat-locked in magnitude |
| Tort MI on population rate, 60 s | MI ~ 2e-5, modulation ~1.5 % of mean rate | entropy-of-bins has no sensitivity to a small modulation on a large mean |

Valid alternatives: neighbour-frequency baselines (the SS-EP convention), neighbour-
tempo grids for phase-binned statistics, and for phase-sensitive questions
(latency, slope) a beat-phase origin from the audio.

## 5. Files

`nmedt_pilot.py` (loader, SS-EP check, v1 MI), `nmedt_pilot_v2.py` (coherent
locking vs circular shift: the negative result), `nmedt_pilot_v3.py` (refined
tempo, windowed tagging z, phase-randomized surrogate), `nmedt_pilot_v4_units.py`
(per-unit and unit-PC1 readouts). JSON outputs: `pilot_v3_song{21,28,30}.json`.
