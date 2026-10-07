# End-to-end workflow

The standard operating procedure from idea to amplification. Each stage
lists entry criteria, procedure, exit criteria, and the command that
gates it.

## Module 1: AI protocol (standing)

- Work one scoped phase at a time; present, verify, proceed.
- State evaluation criteria before generating; verify after.
- Anchor to venue guidelines, model papers, and this project's data.
- Report negative results exactly.

## Module 2: Ideation gate

- Entry: a candidate idea.
- Procedure: declaration test, aunt pitch, novelty vs. originality,
  anti-salami, mole-hill scoping; then journal selection (mine the
  reference list for journals appearing twice or more, verify scope,
  turnaround, and open-access, screen for predatory venues).
- Exit: a go decision and a chosen venue with its blinding policy.
- Gate: `/idea-gate`; record a `decision` memory entry.

## Module 3: The Wall

Build order and what "done" means:

| Phase | Done when | Gate |
| --- | --- | --- |
| figures | at least one float or `\includegraphics` | `phase_gate.py` |
| methods | methods section with substance | `phase_gate.py` + `methods.*` |
| results | results section with substance | `phase_gate.py` + `results.*` |
| discussion | discussion with citations and hedging | `discussion.*` |
| conclusion | limitations with pivots | `limits.*`, `conclusion.*` |
| introduction | compound gap + crunchy contributions | `intro.*` |
| front matter | title, abstract, keywords | `title.*`, `abstract.*` |

Run `python3 scripts/phase_gate.py manuscript/main.tex` before each
phase; never draft a later phase while an earlier one is thin.

## Per-phase drafting loop

1. State the section's criteria (its check ids).
2. Draft only that phase.
3. `python3 scripts/paper_check.py manuscript/main.tex --profile draft`.
4. Fix findings; record a `lesson` for anything caught. STOP.

## Module 5: Submission navigation

- Entry: the manuscript passes the submission profile.
- Blinding: set `venue.blinding`; run the scrub if double-blind.
- Cover letter: `/cover-letter`, then `check_cover_letter.py`.
- Reviewer suggestions: from the reference list; never co-authors, close
  colleagues, or same-institution names.
- Rebuttal: after a 7-day cooling-off, `/rebuttal`, then
  `check_rebuttal.py --strict`.
- Gate: `python3 scripts/paper_check.py manuscript/main.tex --profile submission`.

## Module 6: Amplification

After acceptance, `/amplify`: publication date, press release, social
thread, graphical-abstract prompts. Accuracy over hype.

## Module 7: Memory and retrospective

- Record entries at every phase boundary and after every review round.
- After each round: `python3 scripts/self_improve.py`.
- Close the loop: a mistake reaching a reviewer twice becomes a check or
  a standing rule.
