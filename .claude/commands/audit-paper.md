---
description: Full manuscript audit against the Q1 harness.
---

Run the complete deterministic audit and interpret it.

1. `make -f makefiles/ManuscriptHarness.mk harness-verify`
2. `python scripts/paper_check.py manuscript/main.tex --profile submission --format text`
3. Group the findings by module (front matter, intro, methods, results,
   discussion, floats, structure, compliance, language, bibliography).
4. For each finding, state the fix in one line. Do not edit yet.
5. Report: passed checks, failed checks, and the top three risks.

STOP after reporting. Wait for the user to choose what to fix.
