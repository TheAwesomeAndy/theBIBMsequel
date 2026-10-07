---
description: Scoped revision of a single section.
---

Revise only the section named by the user.

1. State the evaluation criteria the revised text must satisfy (the ghost
   verifier), including the check ids that apply to this section.
2. Make the smallest edit that satisfies them.
3. Re-run `python scripts/paper_check.py manuscript/main.tex --profile draft`
   and confirm the targeted findings cleared with no new ones.
4. Show the diff and the before/after finding counts. STOP.
