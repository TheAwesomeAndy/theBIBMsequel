---
description: Draft one Wall phase in build order.
---

Draft exactly one phase of the paper, honoring the Wall order.

1. Run `python scripts/phase_gate.py manuscript/main.tex`.
2. Identify the earliest phase that is still `[todo]`. That is the only
   phase you may draft this turn.
3. State the section's evaluation criteria (relevant check ids).
4. Draft only that phase.
5. Run `python scripts/paper_check.py manuscript/main.tex --profile draft`
   and `python scripts/phase_gate.py manuscript/main.tex`. Fix findings.
6. Record a `lesson` memory entry for anything the gates caught. STOP.
