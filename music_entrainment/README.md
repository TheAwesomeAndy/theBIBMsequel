# music_entrainment — Week 1 review of "Rhythmic entrainment in ARSPI-Net"

Separate side project from the BIBM/SPL affective line. Nothing here touches the
`BIBM` or `dissoAdventureExperiments` repositories; the pilot imports
`chapter5Experiments/experiment_zero.py` read-only (set `--repo`).

* `WEEK1_ASSESSMENT.md` — literature-backed assessment of the proposal, pilot findings, redesign, decision points.
* `docs/week1_proposal.txt` — the proposal as retrieved on 2026-11-08.
* `docs/lit_*.md` — three literature reviews (entrainment-vs-evoked debate; open music-EEG datasets; reservoir/EEG novelty check).
* `pilot/` — audio-free feasibility pilot on NMED-T songs 21, 28, 30 and its aggregate outputs (`pilot/RESULTS.md`).

Data: NMED-T (Stanford Digital Repository, CC BY; purl.stanford.edu/jn859kj8079) is
downloaded to `/home/user/data_local/nmedt/` and never committed. From the cloud
sandbox the Stanford hosts need the public emSign Root CA - G1 appended to the proxy
CA bundle (see `WEEK1_ASSESSMENT.md` §4). Audio is not distributed with NMED-T.

Status: assessment complete; Phase 2 (calibration harness, Weineck 2022 dataset,
all ten NMED-T songs) awaits approval.
