---
name: wall-drafting
description: Draft the manuscript in Wall build order (figures, methods, results, discussion, conclusion, introduction, front matter) one phase at a time. Use for all drafting.
---

# Wall Drafting

Use whenever drafting a section.

## Procedure
1. `python scripts/phase_gate.py manuscript/main.tex`.
2. Draft only the earliest `[todo]` phase.
3. Verify with paper_check (draft profile) and re-run the phase gate.
4. Record a `lesson` for anything the gates caught.

## Verification
The phase gate reports no ordering violation and the drafted phase is now
`[done]`.

## Stop point
Present the drafted phase and stop for verification.
