# Week 1 "Rhythmic entrainment in ARSPI-Net": research assessment and redesign

Date: 2026-11-08. Author role: lead developer / primary researcher review.
Inputs: the Week-1 proposal page (saved as `docs/week1_proposal.txt`), three
literature reviews (`docs/lit_entrainment_debate.md`, `docs/lit_datasets.md`,
`docs/lit_reservoir_novelty.md`), and a same-day feasibility pilot on three
NMED-T songs (`pilot/RESULTS.md`). Evidence labels: **[measured]** = run here
today; **[lit]** = located and read via the tools; **[inference]** = my reading.

---

## 1. Bottom line

The proposal asks a real question in a live debate and reuses the right code,
but as written it would not survive review, and two of its pilot-stage
assumptions fail on the actual data.

* The headline inference ("real > evoked twin ⇒ beat-locked structure beyond
  linear superposition ⇒ evidence relevant to entrainment") is already known to
  be non-diagnostic in this field **[lit]**. Residual structure after a linear
  TRF shows nonlinearity or misfit, not oscillators; Doelling & Assaneo 2021 say
  outright that a leaky integrate-and-fire model "is both an oscillator and a
  model of evoked responses", and Damsma et al. 2025 (Cerebral Cortex) show that
  the tempo-selective enhancement and spectral signatures are reproduced by both
  model classes, with EEG phase concentration falling between them.
* The primary null (circular shift) and the twin's surrogate (phase
  randomisation) are both invariant for magnitude statistics on a periodic beat
  grid. Measured today: the phase-randomised drive gives the same per-unit
  tagging as the real drive (82 % vs 77 % of units, song 28) **[measured]**.
* The proposed readout (Tort MI on the population rate) has no sensitivity:
  units lock at different phases and cancel in the mean (group z 1.9 vs 24.7 in
  the drive). Per-unit rates keep the information (77 % of units; unsupervised
  unit-PC1 z 22.2) **[measured]**.
* The ten-song cross-tempo slope rests on songs with very different tagging
  strength (none detectable for the 56 BPM song at group level) and a tempo
  range (factor 2.7) narrower than Doelling 2019 (factor 8) or Damsma 2025
  (factor 4), where the same test gave opposite verdicts **[measured, lit]**.
* The audio is not distributed with NMED-T, so the mTRF twin needs ten tracks
  purchased and aligned **[lit, verified on the record]**.

What is genuinely new and worth keeping **[lit]**: no one has put a fixed
spiking reservoir on music EEG; no one has used a linear-model-matched surrogate
of the TRF residual as a null for a nonlinear readout; no one has published a
beat-locked TRF or phase-lag analysis on NMED-T. The "reservoir as detector"
framing has a citable lineage (Ibáñez-Soria 2017 Chaos, 2019 PLoS ONE;
HeteroRC 2026 preprint; reservoir-observer literature) but has not been applied
here.

The publishable path is a reframe: **a characterised nonlinear readout
benchmarked against linear and trivial-nonlinear detectors on naturalistic
music, with an explicit evoked-model baseline**, where the result is publishable
whether or not the reservoir wins. Section 6 lays it out.

---

## 2. What the proposal gets right

* Choosing a stimulus-locked, falsifiable question over another classifier.
* Insisting the reservoir be calibrated on synthetic drives (gain, group delay)
  before interpreting brain data, and pre-registering β, θ, ρ and seeds.
* Treating H1b (does the reservoir add anything?) as a test, not a premise. The
  pilot says this was the correct instinct.
* The code-reuse map is accurate: every module named exists at the stated path
  **[measured]**, and `experiment_zero.LIFReservoir` runs 282 s of continuous
  256 Hz drive in 0.9 s at firing rate 0.126, the SHAPE operating point.
* Noting the two reservoir variants (experiment_zero vs the Chapter 6 sparse,
  hard-reset variant) and the 2.09 Hz low-pass corner inside the beat range.

---

## 3. State of the literature (compressed; details in the three review files)

**The debate.** Frequency tagging (Nozaradan 2011, 2012; Lenc 2018 PNAS) vs the
evoked-superposition critique (Novembre & Iannetti 2018 PNAS; Rajendran &
Schnupp 2019). Doelling et al. 2019 PNAS argued for an oscillator from phase
concentration across note rates (MEG); Damsma et al. 2025 found EEG phase
concentration between the oscillator and a fixed-kernel evoked model and
concluded for mixed contributions. The consensus recommendation (Doelling &
Assaneo 2021; Zoefel 2018; Haegens & Zion Golumbic 2018; Damsma 2025) is to stop
asking "oscillation or not" and compare explicit, falsifiable models with an
evoked model as baseline, on the same data, out of sample.

**Diagnostic tests and their status** (from `lit_entrainment_debate.md` §2):

| Test | Status |
|---|---|
| Phase lag vs rate / phase concentration | partially non-diagnostic as a binary test; dispersive linear kernels keep constant phase in band; two labs got opposite verdicts |
| Residual after linear TRF | untested in the beat literature; diagnostic only for "beyond this linear model", not "not evoked" |
| Forward entrainment after offset | contested both ways; inapplicable to continuous full songs |
| Tempo-selective enhancement | non-diagnostic (Damsma 2025) |
| Frequency-tagged amplitude | non-diagnostic (both accounts predict the spectra) |
| Missing-pulse energy (Tal 2017) | rules out linear superposition only |
| Explicit model comparison with an evoked baseline | the endorsed path |

**Naturalistic music specifically.** Tracking is weaker than for speech (Zuk
2021); spectral flux beats envelope (Weineck 2022); frequency-tagged beat effects
are small (Nave 2025, N = 152 registered report, imagery effect did not
replicate). Kaneshiro 2020 (NMED-H, not NMED-T) found beat-frequency coherence
does not explain time-domain inter-subject correlation.

**Reservoirs and EEG.** Classification uses dominate (Fourati 2018 on DEAP/SEED;
Moinnereau 2018 vowels; Soures & Kudithipudi 2019 review). Detector uses:
Ibáñez-Soria 2017/2019 (generalized synchronisation; SSVEP non-stationarity),
HeteroRC 2026 (reservoir states recover non-phase-locked EEG structure linear
decoders miss). No reservoir on music/rhythm EEG. Oscillatory reservoirs model
rhythm perception/production without EEG (Yuan et al. 2025; Kawai et al. 2024).

**Datasets** (ranked for a beat-locked TRF with tempo as a factor):

1. Weineck, Wen & Henry 2022 eLife, OSF y5xhs: real music time-stretched to
   13 tempi (1–4 Hz), raw EEG for 34 analysed participants, released envelope,
   derivative, beat onsets and spectral flux per trial; 32 channels; no audio.
2. NMED-T: 128 ch, 20 participants, 10 full songs at 10 steady tempi, tapping;
   audio not distributed (ASINs), no beat annotations; raw files carry a click
   every 60 s for alignment once audio is bought.
3. Damsma et al. 2025, OSF ghp94 + UvA DARE: 5 tempi, 64 ch, waveforms and
   IOIs included, N = 12, percussive pattern, the cleanest positive control.

Complementary: Lenc 2020 (audio recorded into EEG channel 69; one tempo),
OpenMIIR (8 tempi, wav + beat times, 7–16 s fragments), MUSIN-G (open WAVs,
128 ch, no tempo design).

---

## 4. Pilot findings (full table in `pilot/RESULTS.md`)

Three cleaned NMED-T songs, canonical `experiment_zero` reservoir, group-PC1
drive and each participant's single-listen projection; windowed Welch tagging z
against neighbouring frequencies (the SS-EP convention).

| | song 21 (56 BPM) | song 28 (120 BPM) | song 30 (150 BPM) |
|---|---|---|---|
| nominal tempo vs EEG-refined | +0.5 % (weak peak) | +0.005 % | +0.012 % |
| drive, best harmonic z | 0.8 | 24.7 (4× tempo) | 21.4 (2× tempo) |
| reservoir population rate z | 1.7 | 1.9 | 2.5 |
| reservoir unit-PC1 z | — | 22.2 | 18.2 |
| units with z > 1.96 (chance 0.025) | — | 0.77 | 0.64 |
| single listens detected, drive / pop-rate | 0.10 / 0.00 | 0.70 / 0.05 | 0.50 / 0.15 |
| phase-randomised drive, unit tagging | — | 0.82 (same as real) | 0.65 (same as real) |

Readings: (i) beat-related tagging is present and strong at the group level for
two of three songs, mostly at metrical subdivisions, absent for the slow
bass-heavy song; (ii) the population-rate readout destroys it, per-unit readouts
keep it; (iii) in the canonical configuration the reservoir does not exceed its
own linear input; (iv) the two surrogates in the proposal are not nulls for
magnitude statistics on a periodic grid.

Data-quality facts also established: tap times are quantised to 4096/44100 s
(92.9 ms), so tapping verifies tempo (all ten within ~1 %) but cannot supply
beat phase; cleaned files are MATLAB v7.3; the Stanford hosts need the emSign
root added to the sandbox CA bundle.

---

## 5. Issues with the design as written, ranked

**Fatal to the stated claim**

1. "Beyond linear superposition" cannot be read as evidence about entrainment;
   the nonlinear-evoked alternative (adaptation, amplitude-dependent latency,
   rectification; Rajendran 2017/2020; Drennan & Lalor 2019) is the live null,
   not linear superposition. The claim must be re-scoped to "beat-locked
   structure not captured by a maximal linear encoding model", and then compared
   against an adaptive-evoked model, not a ridge TRF.
2. Circular-shift and phase-randomisation nulls are non-diagnostic for MI and
   locking magnitude on a periodic grid (measured). Use neighbour-frequency or
   neighbour-tempo baselines; use circular shifts only for phase-sensitive
   statistics or aperiodic grids.
3. A single ridge-mTRF residual with one listen per song is expected to carry
   beat-locked linear leakage (feature misspecification: envelope vs spectral
   flux vs onsets; lag truncation; shrinkage; non-stationarity over 4.5 min).
   "Real > twin" is then nearly guaranteed and says nothing. The linear family
   must be maximised (multi-feature, per-channel, nested CV over λ and lags,
   leave-one-song-out), and a shifted-residual surrogate reported alongside.

**Major**

4. Population-rate MI is the wrong readout (measured). Minimum: per-unit
   beat-cycle histograms, then a fixed unsupervised combination (unit-PC1 or
   summed per-unit z), with the combination rule pre-registered.
5. The phase-lag-vs-tempo slope: already yields opposite verdicts in two labs,
   range here is factor 2.7, one song per tempo confounds tempo with
   instrumentation, spectral flux, lyrics and familiarity, and the 56 BPM song
   shows no measurable tagging. Report descriptively at most.
6. The reservoir adds no diagnostic power by itself; a real-vs-twin difference
   in its output is a statement about the input's residual. Reviewers will ask
   why PLV/ITPC or a bispectrum of the residual does not suffice, and will want
   a shuffled-weights reservoir and input-free runs to exclude self-generated
   phase concentration. The pilot already shows the linear statistic is at
   least as sensitive as the unit readout.
7. Audio must be purchased and aligned; different masters/trims against the lab
   mono conversion are a real risk; alignment must be verified with the 60-s
   click grid and envelope cross-correlation.
8. Stimulus-side controls are mandatory (López-Madrona 2026): beat-locked
   residual structure can come from features absent from the TRF. Drive the
   reservoir with the audio features themselves.

**Minor**

9. The 2.09 Hz low-pass corner sits inside 0.93–2.5 Hz; real and twin pass the
   same filter, but per-tempo gain must be calibrated and reported.
10. Multiple comparisons across 256 units, four harmonics, 125 electrodes and
    descriptors; the pre-registration must name one drive, one readout and one
    or two statistics.
11. Power: n = 20, 80 % power for d_z ≈ 0.66 only. Nave 2025 found 0.03 µV
    beat effects at N = 152.
12. Table 1 tempi are exact (120.006, 150.018 BPM measured from EEG) but 56 BPM
    is uncertain (+0.5 %, weak peak); coherent full-song statistics need tempo
    to 0.1 % and should be windowed instead.

---

## 6. Redesign: a programme with publishable output either way

### Paper A (guaranteed output): "A fixed spiking reservoir as a beat-locking detector for naturalistic-music EEG: calibration and detector-power comparison"

Question: does a pre-characterised, untrained LIF reservoir detect beat-locked
structure in single-listen music EEG, and how does its sensitivity compare with
linear tagging and with trivial nonlinear statistics?

Design:
* Datasets: NMED-T (128 ch, 10 songs) for the main analysis; Weineck 2022
  (13 tempi, released features) for the tempo manipulation; Damsma 2025 as a
  positive control with known evoked structure.
* Detectors, all with the identical neighbour-frequency null and windowing:
  (L) linear tagging of the drive; (N1) PLV/ITPC of the residual at beat
  harmonics; (N2) tagging of the Hilbert envelope and of the squared residual;
  (N3) amplitude-binned TRF gain (Drennan & Lalor); (R) reservoir per-unit
  readout, canonical parameters; (R') reservoir with shuffled weights and
  ρ ∈ {0, 0.3, 0.9, 1.2}.
* Calibration on synthetic drives: beat-locked sinusoid plus pink noise at
  matched SNR; evoked kernel convolved with beat impulses; GrFNN/Wilson–Cowan
  oscillator; report per-tempo gain, group delay and detection curves
  (sensitivity vs SNR) for each detector.
* Outcome metrics: fraction of single listens detected at matched false-positive
  rate; subject-level paired contrasts with participant-cluster bootstrap CIs
  (reuse `participant_bootstrap_ci`); equivalence bounds pre-registered.

Publishable results either way: if R ≤ L on real data (what the pilot suggests),
the paper is a calibrated negative with a documented reason (thresholding and
leak act as a low-pass, the readout cancels phase-diverse units) and a
recommendation; if R > N1–N3 on the residual, it is a positive methods result.
Both include the first beat-locked TRF and single-listen tagging reliability on
NMED-T, which the data review found nobody has published.

Venues: eNeuro, NeuroImage, J Neural Eng, PLoS Comput Biol (methods), or IEEE
TNSRE / EMBC for the instrument-calibration framing.

### Paper B (conditional on A, and on the audio): "Does naturalistic-music EEG contain beat-locked structure beyond a maximal linear encoding model, and is it better explained by adaptive-evoked or oscillator dynamics?"

Only if A shows residual beat-locked structure by any detector: fit an
adaptive-evoked family (IOI/habituation kernels, amplitude-dependent latency) and
an oscillator family (Wilson–Cowan / GrFNN) to the residual, compare out of
sample, anchor to NMED-T tapping consistency and enjoyment ratings. This is the
contribution the field says it wants. It does not need the reservoir; the
reservoir enters as one readout whose sensitivity A has characterised.

### Mapping to the stated goals

1. *Publishable results*: Paper A is near-certain; its negative branch is as
   publishable as its positive one if the calibration is done properly.
2. *High-impact literature*: Paper B, if the residual structure exists. Honest
   odds from the literature: modest; Nave 2025 and Zuk 2021 say beat effects in
   EEG are small and naturalistic tracking is weak.
3. *ARSPI-Net as a necessity for clinical affective research*: this project
   cannot deliver that. It is auditory cognitive neuroscience on healthy
   listeners with no affective or clinical variable, and the pilot shows the
   canonical reservoir does not out-detect its own linear input. What it can
   deliver is the "pre-characterised measurement operator" credential
   (calibration curves, seed/ρ stability, known filtering) that the BIBM paper
   and the SPL submission already lean on. The necessity argument for affective
   work has to come from the sequel line: subject-covariance entanglement,
   leak-free fixed projections, and perturbation characterisation on SHAPE and
   DEAP. I would not let Week 1 displace that.

### The "toward consciousness and meditative states" arc

Not supportable from this design. Enjoyment/familiarity in NMED-T are
near-floor for familiarity (means 1.0–2.2 of 9) and modest for enjoyment
(3.1–6.8), one rating per song; any correlation with a reservoir index would be
exploratory at best. Rhythmic-practice EEG datasets with audio were not found.
Park it.

---

## 7. Decisions that are yours

1. **Scope**: run Paper A as framed above, or keep the original H1/H1b claim
   (not recommended; expect the "known non-diagnostic rehash" review).
2. **Dataset order**: start with Weineck 2022 (no purchase, 13 tempi, features
   released) and treat NMED-T as the 128-channel replication; or buy the ten
   NMED-T tracks (about USD 10–15) and start there as proposed.
3. **Where the code lives**: I have committed the pilot, results and reviews
   under `music_entrainment/` on the designated branch of `theBIBMsequel`
   because that repository was the only candidate not scoped to the BIBM camera-
   ready; say the word and I move it to a new repository.
4. **Pre-registration content**: one drive (group RCA/PC1 component or a
   fronto-central cluster), one readout (per-unit beat-cycle histogram with a
   fixed combination), two statistics (neighbour-frequency tagging z; residual
   PLV), nulls as in §5, equivalence bounds (e.g. ±0.2 subject-level z).
5. **Reservoir variant**: experiment_zero as primary and the Chapter 6 variant
   as a robustness row (agree with the proposal), with ρ ∈ {0, 0.3, 0.9, 1.2}
   and shuffled weights as instrument controls.

## 8. Next phase (needs your approval before I run it)

Phase 2, about one working day on this box, no purchase needed:
1. Download Weineck 2022 raw EEG + feature files from OSF y5xhs; verify licence
   and feature sampling rate; load two participants.
2. Calibration harness: synthetic beat-locked, evoked-kernel and oscillator
   drives at NMED-T noise levels through (L), (N1)–(N3), (R), (R'); detection
   curves vs SNR and tempo; per-tempo gain and group delay of the reservoir.
3. Extend the NMED-T pilot to all ten songs (7 GB; about 25 min download, a few
   minutes compute) with the per-unit readout and neighbour-tempo nulls, and
   write the single-listen reliability table nobody has published.
4. Draft the pre-registration from §7.4 for your edit.
Verification criteria: every detector recovers the planted synthetic structure
at the expected SNR; the real-data table reproduces the three-song pilot within
bootstrap CIs; no restricted data or audio in the repository.
