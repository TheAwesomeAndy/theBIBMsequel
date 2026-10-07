# Q1 pre-submission checklist

Run these gates in order. Every command must pass before submitting.

## Automated gates
- [ ] `make -f makefiles/ManuscriptHarness.mk harness-verify`
- [ ] `python3 scripts/paper_check.py manuscript/main.tex --profile submission`
- [ ] `python3 scripts/phase_gate.py manuscript/main.tex`
- [ ] `python3 scripts/check_cover_letter.py submission/cover_letter.md --journal "<name>"`
- [ ] `python3 scripts/check_rebuttal.py submission/rebuttal.md --strict` (revision rounds)
- [ ] `make -f makefiles/ManuscriptHarness.mk harness-tests`

## Configuration
- [ ] `venue.blinding` set correctly in `harness/papercheck.toml`
- [ ] `venue.human_subjects` set
- [ ] blinding scrub done if double-blind

## Human judgment
- [ ] one reusable idea, passes the declaration test
- [ ] passes the aunt pitch
- [ ] genuine novelty, not just originality
- [ ] not salami-sliced
- [ ] every discussion paragraph follows the RCI order and quality
- [ ] baselines are genuinely strong, not straw men
- [ ] limitations are honest, not decorative
- [ ] suggested reviewers exclude co-authors, colleagues, same institution

## Memory
- [ ] `python3 scripts/self_improve.py` reviewed
- [ ] lessons recorded, addressed entries resolved
