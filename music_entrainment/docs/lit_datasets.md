# Open EEG/MEG datasets for beat-locked TRF analysis (audio envelope/onsets -> EEG)

Scope: datasets where neural data AND stimulus audio (or at least precise beat/onset timing) are downloadable, so a forward TRF can be fitted and beat-locked phase lag can be examined as a function of tempo. Verified items were read from the actual repository record, README, or paper full text (fetched 2026-11-08). Anything not confirmed from a fetched page is tagged **[unverified]**.

Access note: `purl.stanford.edu` / `stacks.stanford.edu` currently serve an incomplete TLS chain (leaf signed by "InCommon Intermediate CA - DVG2C" with no intermediate sent). Fetching required appending the public intermediate (`http://repository.emsign.com/certs/EEEMIncommonDVG2C.crt`) to the CA bundle. Downloads via wget/curl may need the same fix.

---

## 1. NMED-T (Naturalistic Music EEG Dataset - Tempo)

Record: https://purl.stanford.edu/jn859kj8079 (DOI 10.25740/jn859kj8079). Paper: Losorelli, Nguyen, Dmochowski, Kaneshiro, ISMIR 2017, pp. 339-346 (https://archives.ismir.net/ismir2017/paper/000198.pdf). Licence: CC BY 3.0 on the record; the paper itself is CC BY 4.0.

### Confirmed facts (PURL XML metadata, README PDF, ISMIR paper)

| Item | Value |
|---|---|
| Participants | 20 right-handed adults, 18-29 y (mean 23), 6 female; 17 with musical training (mean 8.4 y). Raw files numbered 1-23 with 1, 18, 22 omitted; README gives the raw-to-clean mapping. |
| Stimuli | 10 full-length commercial songs, 4:31-4:58, electronically produced beats, duple meter, 55.97-150 BPM (0.93-2.50 Hz). Tempo computed with Ellis (2007) beat tracker, perceptually validated by four musicians. Presented mono via Genelec 1030A speakers at 73-78 dB; random order; two recording blocks. |
| EEG system | EGI GES300 / Net Amps 300, 128 electrodes, vertex reference, 1 kHz, 24-bit. |
| Cleaned release | 0.3 Hz Butterworth HP, 59-61 Hz notch, 50 Hz Chebyshev LP, downsampled x8 to **125 Hz**; bad electrodes and transients (>4 SD) imputed from neighbours; electrodes 1-124 + vertex row -> **125 rows**, average-referenced; aggregated per song as 125 x T x 20. |
| Files (55, 39.7 GB total) | `song21_Imputed.mat` ... `song30_Imputed.mat` (10 files, 657-722 MB; variables `dataSS`, `subsSS`, `fs`=125); `PP_R_raw.mat` (40 files, 753-862 MB; `X` 129 x T, `DIN_1`, `fs`=1000); `behavioralRatings.mat` (20 x 10 x 2, familiarity/enjoyment 1-9); `TapIt.zip` (aggregated `allTappedResponses` 20x10 cell of tap times, `allSongOrders`, per-trial `.txt`); `participantInfo.mat`; `Code.zip` (illustrative analyses, `parseDIN.m`, electrode map); `NMED-T_README.pdf`. |
| Tapping | Separate listen, 35-s excerpts (1:00-1:34 + 1 s silence), Tap-It iOS app, ~15 ms (SD 5 ms) tap latency. |
| Audio | **Not distributed.** Record abstract: "Stimulus audio is not published, but metadata links are provided." The "metadata links" are the Amazon ASINs in Table 1 of the paper (songs were purchased from Amazon). |
| Beat/onset annotations | **None distributed.** No envelope, spectral flux, or beat-time files. The only tempo information is the per-song BPM/Hz table (paper Table 1) and the trigger/song/duration table in the README. Fig. 1 of the paper shows envelope spectra computed with MIR Toolbox 1.5 at 125 Hz, but the vectors are not released. |
| Timing/alignment | Raw files: trigger 21-30 = song onset from Presentation software; trigger 128 = click embedded in the 2nd audio channel at 1 s and every 60 s thereafter (not played to participants); 1-9 ratings; 10/11 key presses. Cleaned files are already epoched to click-corrected onsets. |

Song table (paper Table 1; README trigger numbers):

| Trig | # | Title | Artist | ASIN | BPM | Hz | Length |
|---|---|---|---|---|---|---|---|
| 21 | 1 | First Fires | Bonobo | B00CJE73J6 | 55.97 | 0.933 | 4:38 |
| 22 | 2 | Oino | LA Priest | B00T4NHS2W | 69.44 | 1.157 | 4:31 |
| 23 | 3 | Tiptoes | Daedelus | B011SAZRLC | 74.26 | 1.238 | 4:36 |
| 24 | 4 | Careless Love | Croquet Club | B06X9736NJ | 82.42 | 1.374 | 4:54 |
| 25 | 5 | Lebanese Blonde | Thievery Corporation | B000SF16MI | 91.46 | 1.524 | 4:49 |
| 26 | 6 | Canopee | Polo & Pan | B01GOL4IB0 | 96.15 | 1.603 | 4:36 |
| 27 | 7 | Doing Yoga | Kazy Lambist | B01JDDVIQ4 | 108.70 | 1.812 | 4:52 |
| 28 | 8 | Until the Sun Needs to Rise | Rufus du Sol | B01APT6JKA | 120.00 | 2.000 | 4:52 |
| 29 | 9 | Silent Shout | The Knife | B00IMN40O4 | 128.21 | 2.137 | 4:54 |
| 30 | 10 | The Last Thing You Should Do | David Bowie | B018GS2A46 | 150.00 | 2.500 | 4:58 |

### Published analyses of NMED-T (and related claims to check)

| Work | Dataset | One-sentence finding |
|---|---|---|
| Losorelli et al. 2017 (ISMIR) | NMED-T | Mean-electrode and PC1 spatial filters of participant-averaged EEG show low-frequency SS-EP peaks at beat-related frequencies (PC1 clearer); many PC1 spectra also show 5-10 Hz peaks regardless of tempo; tapping is bimodal (tempo-octave split) for the two slowest songs. No phase-lag analysis. |
| Kaneshiro & Dmochowski 2018 (ISMIR LBD, "Evidence of beat entrainment in maximally reliable cortical responses to natural music") | NMED-T | RCA spatial filtering followed by inter-subject coherence shows statistically significant coherence at beat-related frequencies (abstract only; PDF not located). |
| Kaneshiro & Dmochowski 2017 (SMPC poster "Exploring the representation of musical beat in ongoing EEG using component analysis"); Losorelli, Kaneshiro, Berger 2016 (ICMPC poster "Representation of musical beat in scalp-recorded EEG responses: a comparison of spatial filtering techniques") | NMED-T (presumed) | Titles only, from the CCRMA publications page; no abstracts fetched. **[unverified content]** |
| **Kaneshiro, Nguyen, Norcia, Dmochowski, Berger 2020, NeuroImage 214:116559** | **NMED-H, not NMED-T** (N=48, Hindi pop songs, intact / measure-shuffled / reversed / phase-scrambled) | ISC and stimulus-response correlation with the envelope are significant for intact and shuffled music; time-domain correlation is highest for measure-shuffled versions; frequency-domain coherence peaks at beat-related frequencies but peak magnitudes do not explain the time-domain correlations. The PURL record of NMED-H lists this paper as its reference. |
| Pandey, Ahmad, Miyapuram, Lomas 2021 (IEEE BIBM) "Predicting dominant beat frequency from brain responses" | **MUSIN-G**, not NMED-T | Band-power features at beat-hierarchy frequencies (eighth-note band most discriminative) classify songs into dominant-beat groups at 70%/56% (binary/ternary) from 1-5 s windows. |
| Pandey, Sharma, Miyapuram, Subramanian, Lomas 2022 (ICASSP) | NMED-T + MUSIN-G | Classifiers trained on the first 20 s of each song identify the song from later segments; beta/gamma band power is most informative; listener-specific responses. |
| Ramirez-Aristizabal et al. 2022 (arXiv, EEG2Mel) | NMED-T + NMED-H | CNN regressors from 1-s EEG windows reconstruct mel spectrograms (81% classifier accuracy; 85% 2AFC listener identification); implies the authors obtained and aligned the audio themselves (method not stated). |
| Postolache et al. 2024/ICASSP 2025 (latent diffusion) | NMED-T (124 channels, 9 songs train / 1 held out) | EEG-conditioned ControlNet beats a convolutional baseline on embedding metrics; generalisation to the held-out track is limited; audio sourcing not stated. |
| Ahmadi et al. 2025 (EMBC) | NMED-T | GNNs classify enjoyment vs. non-enjoyment from EEG at 87%. |
| Noguchi et al. 2026 (arXiv 2603.03190) | NMED-T (128 ch, 3-s segments, EEG shifted 200 ms) | Pretraining EEG encoders on acoustic and expectation ANN representations raises song identification to 0.887 vs 0.823; audio sourcing not stated. |
| Qing et al. 2026 (arXiv 2606.04040) | NMED-T/H | Channel-preserving alignment improves within-subject 50-way identification from 0.388 to 0.485 and CLAP similarity of generated audio. |
| Katthi & Ganapathy 2021 (Deep CCA, IEEE TNSRE) | "music" EEG dataset, named only as NMED in search snippets **[unverified which]** | Deep correlation models raise stimulus-response correlation ~29% over linear methods for music. |
| UAA student capstone 2024 (not peer-reviewed) | NMED-T cleaned files | CNN/MLP tempo regression from RCA-filtered EEG works near 90 BPM and degrades at extremes with octave errors. |

Not found for NMED-T: any published beat-frequency phase-lag-versus-tempo analysis, any beat-locked TRF, or any released audio features. Stober, Pratzlich & Muller 2016 "Brain Beats" (tempo extraction from EEG) used OpenMIIR, not NMED-T.

---

## 2. Other NMED datasets (Stanford Digital Repository; all Matlab; audio status from the PURL abstracts, READMEs, and file lists)

| Dataset | PURL | N / EEG | Stimuli | Files / size / licence | Audio or features distributed? |
|---|---|---|---|---|---|
| NMED-H v2.3.0 (Hindi) | sd922db3535 | 48 adults; EGI GES300 128 ch; raw release already 0.3-50 Hz filtered and downsampled to **125 Hz** (97 recordings in 4 zips, ~2 GB each); clean 125 electrodes x time x 12 trials per stimulus/listen (32 files); RCA outputs (4 conditions) | 4 Bollywood songs x 4 conditions (intact, measure-shuffled, phase-scrambled, reversed), ~4.5 min each; 12 participants per stimulus, 2 listens | 16 downloadable items, 21.9 GB; CC BY 3.0 | **No audio.** README gives iTunes album IDs plus start/end trim sample indices (fs 44100) and Matlab code to rebuild the trimmed intact stimuli from purchased audio; click triggers every 60 s. No beat/onset files. Songs are not steady-tempo-designed. |
| NMED-M (Minimalism) | kt396gb0630 | 30 adults; dense-array EGI; clean 5 files (~1.09 GB each), raw 30 files (~0.9-1.0 GB each) | Reich *Piano Phase* (Tilles/Neimann 1987 recording), Matt Winn remix, and three manipulations (Abrupt Change, Segment Shuffle, Tremolo); continuous engagement ratings | 40 files, 34.4 GB; CC BY 4.0 | **One stimulus distributed:** `TremoloStimulus.wav` (26.9 MB). Others are commercial recordings, not distributed. |
| NMED-E (Elgar) v3 | pp371jh5722 | 24 adults; 128-ch EEG; raw 24 files (~0.57-0.78 GB), clean 4 files, `RawPhysio.zip` | Elgar Cello Concerto mvt. I (du Pre 1965) plus phase-scrambled envelope-matched control; 60-s pink-noise baselines; continuous engagement | 33 files, 19.2 GB; CC BY 4.0 | No audio in file list (commercial recording). Reference paper: Kaneshiro et al. 2024, Eur J Neurosci (ISC tracks musical tension). |
| NMED-RP (Rhythm Pilot) | rz763kn3821 | 5 adults; EGI 128 ch; 15 recordings (`S0x_E0y.mat`, 616-701 MB) | 16 x 30-s excerpts: Bollywood music in four rhythmic categories (Even, Polyrhythm, Swing, Syncopated), 3 real-world excerpts + 1 "plain" rhythm per category; ratings of enjoyment, complexity, ease of finding the beat | 15 files, 9.9 GB; CC BY 3.0 | No audio in file list. Analysed in Appaji, Dmochowski & Kaneshiro 2020 ISMIR LBD (pulse clarity vs. ISC: r = -0.8). |
| NMED-ST | - | Not found on the CCRMA/MERI datasets page or in SDR; appears not to exist. | | | |

Related Stanford item: "EEG-Recorded Responses to Short Chord Progressions" (purl js383fs8244) - not rhythm-focused, not examined.

---

## 3. Alternative open datasets with audio or precise stimulus timing

Legend: Audio = stimulus waveform downloadable; Feat = stimulus feature vectors (envelope/onsets) downloadable; Raw = unprocessed EEG available.

| # | Dataset (paper) | N | Ch / fs | Stimulus type | Tempo design | Audio / Feat / Raw | Licence | URL | Size | Verdict for beat-locked phase-lag vs tempo |
|---|---|---|---|---|---|---|---|---|---|---|
| A | **Weineck, Wen & Henry 2022 eLife** (OSF Y5XHS) | 37 recorded, 34 analysed | 32 actiCap, 1000 Hz raw (BrainVision .eeg/.vhdr/.vmrk; triggers 40/80 segment start/end, 120/180 repetition) | Instrumental real music (techno, rock, blues, hip-hop...), 72 segments 8.3-56.6 s time-stretched | **13 tempi, 1-4 Hz in 0.25-Hz steps** | Audio: no (commercial). Feat: yes - per-trial envelope, envelope derivative, beat onsets, spectral flux (Figure1_SourceData1 "Stims_TRF_z"), plus original tempi per segment. Raw: yes, all 34 analysed subjects (405 files in `rawEEGdata_examples`) | OSF licence set (ID present) but type not read **[unverified]** | https://osf.io/y5xhs/ | several GB | **Best fit.** Multi-tempo by design, features already extracted, TRF/RCA/coherence pipeline published. Risks: no audio (cannot recompute features at other rates or check lags against waveform), 32 channels, feature sampling rate must be checked in the .mat files, trial-to-marker bookkeeping needed. |
| B | **NMED-T** (above) | 20 | 128/125 Hz clean, 1 kHz raw | Full-length commercial songs | 10 tempi 56-150 BPM (steady beats) | Audio: no (ASINs). Feat: no. Raw: yes | CC BY 3.0 | purl.stanford.edu/jn859kj8079 | 39.7 GB | Strong design; audio must be purchased and aligned (see risks below). |
| C | **Damsma, De Roo, Doelling, Bazin & Bouwer 2025 Cerebral Cortex** (OSF GHP94; raw EEG on UvA DARE 10.21942/uva.26233139) | 12 | 64 BioSemi, 512 Hz | Non-isochronous woodblock pattern in duple meter | **5 tempi** (grid 150/200/300/400/600 ms), >900 pattern repetitions per tempo, attended/unattended/tapping | Audio: `soundSigs.mat` + `IOIsFromLog.mat` (waveforms/onsets as .mat, not wav). Raw: at DARE (file list not visible). Clean epochs on OSF | Not stated | https://osf.io/ghp94/ ; https://dare.uva.nl/id/d836f480-3d18-4371-aae7-9708d8c48f18 | small (OSF) | Good controlled multi-tempo set; N=12 and a percussive non-musical pattern; the paper itself reports tempo-dependent shifts of the dominant metrical frequency. |
| D | **Lenc, Keller, Varlet & Nozaradan 2020 Cerebral Cortex Communications** (figshare 11366120) | 32 (16 musicians) | 64 BioSemi + 2 mastoids, 2048 -> 512 Hz; channel 69 = copy of stimulus audio; accelerometers | Pure-tone rhythms on a 200-ms grid (5 Hz events), 57.6-s sequences, regular->degraded and reverse | Single grid tempo (meter 1.25/2.5/5 Hz) | Audio: yes (`stimuli.zip` 1.09 GB incl. cochlear-model output). Raw: yes (letswave .lw6/.mat, ~280 MB/subject) + tapping wavs | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.11366120 | ~10 GB | Excellent for TRF validation (audio co-recorded in EEG), but one tempo. |
| E | **Coulon, Baum, Lenc, Polak & Nozaradan 2025 Sci Rep** (OSF YZ4HD) | 52 (26 musicians) | 64 + 2 mastoids, 512 Hz | Pure-tone rhythms, 200-ms grid, 67.2-s sequences; repetition conditions | Single tempo (beat 1.25/2.5 Hz) | Audio: 15 wav files. Preprocessed EEG (.mat, ~1.3 GB per group); raw on request | Not stated | https://osf.io/yz4hd/ | ~4.7 GB | Large N, audio included, but single tempo and preprocessed only. |
| F | **Barbero et al. 2025 Science Advances** (OSF 9H43J) | 18 | 64 BioSemi, 1024 Hz raw (letswave) | 13 two-interval rhythmic patterns (1:1 to 2:1), 750-ms period, 22.5-s sequences | Single period | Audio: `stim.mat`; cochlear-model responses; Raw EEG + tapping; LICENSE file | See LICENSE file **[unverified type]** | https://osf.io/9h43j/ | ~12 GB | Rhythm-categorisation design, not tempo; raw EEG is a plus. |
| G | **Nave, Hannon, Snyder & RAFT Consortium 2025 registered report** (OSF RPVDE + Harvard Dataverse 10.7910/DVN/ELZFWK) | 152 analysed (13 labs) | Heterogeneous (bdf, BrainVision...), >=1000 Hz | 333-Hz tone AM at 2.4 Hz, 33-36 s, imagined binary/ternary meter | Single rate | Audio: stimulus wavs on OSF. Raw: yes, ~175 GB on Dataverse | OSF/Dataverse default **[unverified]** | https://osf.io/rpvde ; Dataverse DOI above | 175 GB | Huge N but one rate and multi-site hardware; failed to replicate Nozaradan 2011 imagery effects. |
| H | **Cabral-Calderin & Henry 2022 J Neurosci** (Edmond 10.17617/3.HKHZ5V) | 38 x 2 sessions | 64 actiCAP, 1000 Hz raw (BrainVision) | 20-s complex tones FM at 2 Hz (control experiment at 1.5/2/2.5/3 Hz) | Mainly single rate; multi-rate control data presence not confirmed **[unverified]** | Stim parameters per subject (`*_Stim_Main.mat`); Raw: yes (286 files) | Open access | https://edmond.mpg.de (DOI above) | ~50 GB | Henry-lab FM entrainment with raw EEG; not music; one main rate. |
| I | **Di Liberto et al. 2020 eLife "diliBach"** (Dryad 10.5061/dryad.g1jwstqmh; also CNSP) | 20 (10 pianists) | 64 + 2 mastoids, 512 Hz | 10 monophonic Bach piano pieces, 30 trials (~75 min) | Natural tempi per piece; no tempo manipulation | Audio wav: not confirmed on record **[unverified]**; MIDI yes; Feat: `dataStim.mat` at 64 Hz incl. note onsets | Dryad (CC0 default) **[unverified]** | https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqmh | 5.98 GB | Onset-TRF ready; melodies are isochronous-ish piano lines but no tempo factor. |
| J | **Marion, Di Liberto & Shamma 2021 J Neurosci "Music of silence I"** (Dryad 10.5061/dryad.dbrv15f0j; CNSP "Music Imagery") | 21 musicians | 64 ch, released at 64 Hz | 4 Bach chorales, ~30 s, listening + imagery, 11 reps each | Single tempo 100 BPM (metronome) | Audio: yes (`original_stim.zip` 14 MB, metronome on left channel) + MIDI + scores; EEG `ImageryData.mat` 1.5 GB | Dryad default **[unverified]** | https://doi.org/10.5061/dryad.dbrv15f0j | 1.52 GB | Audio-aligned and clean, but one tempo and 64-Hz EEG. |
| K | **OpenMIIR** (Stober et al. 2015 ISMIR) | 10 (P01-P08 v1, P09-P14 v2 metadata) | BioSemi 64 ch, 512 Hz **[unverified from repo]** | 12 well-known fragments 7-16 s (4 with/without lyrics, 4 instrumental), with cue clicks; perception + imagination | Natural tempi **104-212 BPM** (cue bpm per piece in `Stimuli_Meta.v2.xlsx`), 3/4 and 4/4 | Audio: yes (`audio/full.v1`, `full.v2` wavs; cues). Feat: librosa beat onsets in `meta/beats.v*`. Raw: FIF, ~700 MB/subject | PDDL | https://github.com/sstober/openmiir | several GB | Eight distinct tempi with audio and beat times, but very short fragments and small N; usable for coarse lag-vs-tempo check. |
| L | **MUSIN-G** (Miyapuram et al. 2022 Data in Brief; OpenNeuro ds003774) | 20 | EGI HCGSN 128 (129 ch), 250 Hz (some 1000 -> 250) | 12 songs of mixed genres, ~2 min, eyes closed | None (natural tempi; Pandey 2021 estimated them) | Audio: **yes**, `Code/ESongs/1.esh.wav`...`12.esh.wav` (3.2-4.2 MB each) + beeps; Raw: yes (BIDS .set) | CC0 | https://openneuro.org/datasets/ds003774 | moderate | Audio present and open; no tempo design, lower fs; lyrics in 3 songs. |
| M | **Keitel et al. 2025 Ann NY Acad Sci** (OSF 3GF6K) | 21 (20 analysed) | 32 BioSemi, 512 Hz (cleaned .mat per subject/condition) | Mozart K.283 II tonal vs pitch-shuffled atonal, 292 s | Single tempo 46 BPM (dominant 92 BPM, 1.52 Hz) | Audio: yes (`T_full.wav`, `A_full.wav`, tapping segments) | Not stated | https://osf.io/3gf6k/ | ~1.7 GB | Audio-aligned real music, but one tempo. |
| N | **Zalta, Large, Schon & Morillon 2024 Sci Adv** (Dryad 10.5061/dryad.7m0cfxq2g; GitHub DCP-INS/Groove) | 30 MEG | not stated | 12 melodies x 3 versions (incl. `_reg_NM_2Hz`), syncopation manipulated | 2 Hz beat | Audio: yes (GitHub `Stimuli/-wav`, 36 files). Raw MEG: **no** (derived spectra/decoding only) | "free of use" | Dryad DOI above | 4.74 GB | Not usable for TRF (no raw MEG). |
| O | **Simon, Bech, Loquet & Ostergaard 2022/2024** (Zenodo 7500806) | 18 | 64 g.HIamp, 2400 -> 512 Hz preprocessed | Disney excerpts (instrumental / Danish / Finnish lyrics) + audiobooks, 70-s trials | None | Audio: no; Feat: aligned envelopes yes; Raw: no | Not stated | https://zenodo.org/records/7500806 | 5.3 GB | Envelope-TRF ready but no tempo factor, no raw. |
| P | **CantiSong** (Cantisani, Shamma & Di Liberto; Zenodo 14590637) | not stated | CND at 64 Hz | 18 songs: speech, sung, hummed versions (6 singers) | None | Audio: yes (wav with trigger channel), MIDI melody, phoneme alignments | CC BY 4.0 | https://zenodo.org/records/14590637 | 3.4 GB | Good audio alignment; vocal music, no tempo design, 64 Hz. |
| Q | **Shan & Maddox 2024, OpenNeuro ds004356** | 22 | BrainVision raw | 12-s clips, 6 music genres + 6 speech types, 40 trials each, plus clicks | None | Audio: yes (`stimuli/<genre>/*.wav`) | CC0 | https://openneuro.org/datasets/ds004356 | large | ABR-oriented; short clips, no tempo design. |
| R | Daly/Nicolaou (Reading) tempo set: "EEG data during 'peaceful' auditory processing at different tempi" (Nicolaou et al. 2017 Front Hum Neurosci) | 21 | 19 ch (10-20), 1 kHz | 9.5-s "peaceful" clips from Vieillard et al. 2008 + noise versions | **4 tempi: 50/100/150/200 BPM** | Audio: no (clip names given; Vieillard stimuli available from original authors); Raw: yes | CC BY 4.0 | https://researchdata.reading.ac.uk/id/file/4744 (record) | small | Multi-tempo but 9.5-s trials, 19 channels, audio external. |
| S | Daly et al. 2020 Sci Data family: ds002721 (31 subj, 40 film-music 12-s clips, no audio), ds002720 tempo-BCMI (19 subj, generated music, generator code only), ds002722/23/24 (generated affective music), ds002725 EEG-fMRI | 8-31 | 1 kHz | Generated or film music | None (BCMI tempo is user-controlled) | Audio mostly not shipped (generated at runtime) | CC BY 4.0 / CC0 | openneuro.org | varied | Not suitable. |
| T | DEAP | 32 | 32 ch, 512 Hz (128 Hz preprocessed) | 40 one-minute music-video excerpts | None | Audio: no (YouTube links; EULA) | EULA | eecs.qmul.ac.uk/mmv/datasets/deap | - | Not suitable **[unverified: page returned 503; from memory]**. |
| U | Broderick et al. 2018 "Natural Speech" (Dryad 10.5061/dryad.070jc; CNSP CND) | 19 | 128 BioSemi, 512 Hz **[unverified from secondary sources]** | Audiobook speech | n/a | Feat: envelopes yes (CNSP dataStim); Audio: not shipped **[unverified]** | Dryad default | datadryad.org DOI above | 14.8 GB (all sub-sets) | Speech, not music; useful only as a TRF pipeline check. |
| V | Henry, Herrmann & Grahn 2017 PLoS ONE (Dryad 10.5061/dryad.37jg4) | - | - | - | - | **Behavioural data only (38 KB zip); no EEG** | - | - | - | Not usable. |
| W | OpenNeuro ds005876 "Song Familiarity" (29 subj; melody wavs + synchronized audio reconstruction in derivatives, CC0); ds002218 "Auditory and Visual Rhythm Omission" (18 subj, 1000-Hz tone wav, CC0); ds006735 "Chimeric music" (27 subj; stimuli not in top-level listing) | | | | | | | openneuro.org | | Peripheral; none has a tempo factor. |

Not public / no deposit found: Lenc et al. 2018 PNAS (no data statement in PMC full text); Nozaradan 2011/2012/2016 **[unverified, no statements located]**; Henry & Obleser 2012 and Zoefel datasets **[not found]**; Rosso et al. 2023 NeuroImage ERFA (no deposit found); Zuk et al. 2021 (EEG explicitly not shareable; ethics). Sturm et al. 2015 and Cong et al. 2013 naturalistic-music EEG: no public deposits located **[unverified]**. No open MEG dataset of music listening with raw MEG and audio was found (MEG hits on OpenNeuro were speech/word datasets; Zalta 2024 releases derived MEG only).

---

## 4. Datasets with isochronous or steady-beat stimuli at multiple tempi and public timing

| Dataset | Tempi | Stimulus timing available | EEG raw | Comment |
|---|---|---|---|---|
| Damsma et al. 2025 (OSF ghp94 + UvA DARE) | 5 (grid 150-600 ms) | yes (`soundSigs.mat`, `IOIsFromLog.mat`, `check_sound_timings.m`) | yes (DARE) | Non-isochronous duple pattern; tapping too. |
| Weineck et al. 2022 (OSF y5xhs) | 13 (1-4 Hz) | beat-onset vectors and envelopes per trial | yes | Real music, time-stretched. |
| NMED-T | 10 (0.93-2.5 Hz) | tempo per song + click triggers; no beat times | yes | Need purchased audio. |
| OpenMIIR | 8 (104-212 BPM) | librosa beat times + wav | yes (FIF) | 7-16 s fragments. |
| Reading "peaceful" tempo set | 4 (50-200 BPM) | tempo label per trial only | yes | 9.5-s clips, 19 ch. |
| Cabral-Calderin & Henry 2022 | 2 Hz main (1.5-3 Hz control) | FM parameters per trial | yes | FM tones, not beats; multi-rate data presence **[unverified]**. |
| Nave et al. 2025 replication; Lenc 2020; Coulon 2025; Barbero 2025 | single rate each | wav + onsets | yes (Nave, Lenc, Barbero) / preprocessed (Coulon) | Frequency-tagging designs, one tempo. |

---

## 5. Recommendation ranking (purpose: forward TRF from audio envelope/onsets, beat-locked phase lag vs tempo)

1. **Weineck, Wen & Henry 2022 (OSF y5xhs).** Only open dataset with real music at 13 controlled tempi (1-4 Hz), raw EEG for all 34 analysed participants, and the needed stimulus features (envelope, derivative, beat onsets, spectral flux) already released with the per-segment original tempo. The authors' TRF/RCA pipeline (Kaneshiro-style RCA, coherence) is public. **Main practical risk:** the audio itself is not shared (commercial music), so features cannot be regenerated, filter/rate choices are fixed by the released vectors, and only 32 channels; the OSF licence type and the feature sampling rate must be checked in the .mat files before use.

2. **NMED-T (Stanford SDR).** Best channel count (128), 20 participants, ten ~4.7-min songs spanning 0.93-2.5 Hz with steady electronic beats, 1-kHz raw EEG with click-based timing, plus tapping data for perceived tempo. **Main practical risk:** the audio must be purchased (Amazon ASINs) and aligned by hand; only the song-onset trigger and the 60-s click grid are available, so offsets from different masters/releases or lab-side trimming/mono conversion must be verified (e.g., by checking SS-EP phase consistency across the 60-s click segments), and no beat-onset annotations exist (beat tracking must be run on the purchased audio). Licence CC BY 3.0; 39.7 GB.

3. **Damsma et al. 2025 (OSF ghp94 + UvA DARE).** Controlled five-tempo rhythm (150-600 ms grid), 64-channel EEG at 512 Hz, stimulus waveforms/IOIs and tapping included, and the paper already documents tempo-dependent frequency shifts, making it the cleanest positive control for a lag-vs-tempo TRF. **Main practical risk:** N = 12, the stimulus is a percussive woodblock pattern rather than music, raw EEG lives in a second repository (DARE) with no visible licence or file list, and the OSF "Clean data" are epoched rather than continuous.

Runner-up / complementary: Lenc et al. 2020 (figshare; audio literally recorded in EEG channel 69 - ideal for validating TRF latency estimates at one tempo), OpenMIIR (eight tempi with wav + beat times but 7-16 s fragments), Marion 2021 and Keitel 2025 (audio-aligned real music, single tempo), MUSIN-G (open audio and 128 channels, no tempo design).

---

## Sources (fetched)

- NMED-T PURL XML and README: https://purl.stanford.edu/jn859kj8079.xml ; https://stacks.stanford.edu/file/druid:jn859kj8079/NMED-T_README.pdf
- Losorelli et al. 2017 ISMIR PDF: https://archives.ismir.net/ismir2017/paper/000198.pdf
- NMED-H/M/E/RP PURL XML: https://purl.stanford.edu/sd922db3535.xml , kt396gb0630.xml , pp371jh5722.xml , rz763kn3821.xml ; NMED-H README: https://stacks.stanford.edu/file/druid:sd922db3535/NMED-H_v2.3.0_README.pdf
- CCRMA MERI datasets/publications pages: https://ccrma.stanford.edu/groups/meri/datasets.html ; https://ccrma.stanford.edu/groups/meri/publications.html
- Appaji, Dmochowski & Kaneshiro 2020 ISMIR LBD: https://program.ismir2020.net/static/lbd/ISMIR2020-LBD-419-abstract.pdf
- Kaneshiro et al. 2020 abstract (NeuroImage): https://dblp.dagstuhl.de/rec/journals/neuroimage/KaneshiroNNDB20.html and Consensus record; Kaneshiro & Dmochowski 2018 LBD abstract via Consensus
- arXiv 2603.03190 (HTML), 2405.09062 (abs), 2606.04040, 2207.13845 (Consensus abstracts)
- Weineck 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9467512/ ; OSF API listing of https://osf.io/y5xhs/ and ReadMe.pdf
- Damsma 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12461700/ ; https://osf.io/ghp94/ (API) ; https://dare.uva.nl/id/d836f480-3d18-4371-aae7-9708d8c48f18
- Lenc 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC8152888/ ; figshare API https://api.figshare.com/v2/articles/11366120 and README files
- Lenc 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC6094140/
- Coulon 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12789069/ ; https://osf.io/yz4hd/ (API)
- Barbero 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12309689/ ; https://osf.io/9h43j/ (API)
- Nave 2025: https://www.biorxiv.org/content/10.1101/2025.03.13.643168v3.full ; OSF rpvde (API); Dataverse API for doi:10.7910/DVN/ELZFWK
- Cabral-Calderin & Henry 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC8808727/ ; Edmond API doi:10.17617/3.HKHZ5V
- Di Liberto 2020 Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqmh ; Marion 2021 Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.dbrv15f0j ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8412990/
- CNSP resources: https://cnspworkshop.net/resources.html ; CantiSong: https://zenodo.org/records/14590637
- OpenMIIR: https://github.com/sstober/openmiir (README, meta/Stimuli_Meta.v2.xlsx, audio/full.v1)
- MUSIN-G: https://pmc.ncbi.nlm.nih.gov/articles/PMC9679455/ ; OpenNeuro GraphQL file tree of ds003774 v1.0.2 ; GitHub mirror
- OpenNeuro sweep (469 EEG + 68 MEG dataset names via GraphQL) and READMEs of ds002218, ds002720, ds004356, ds005876, ds006735, ds003570, ds002721-2724
- Zalta 2024: https://datadryad.org/dataset/doi:10.5061/dryad.7m0cfxq2g ; https://github.com/DCP-INS/Groove
- Simon et al.: https://zenodo.org/records/7500806 ; Keitel 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11998481/ and OSF 3gf6k (API)
- Reading tempo dataset README: https://researchdata.reading.ac.uk/id/file/4744
- Zuk 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8480853/ ; Henry 2017 Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.37jg4 ; Broderick Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.070jc
