---
name: manuscript-audit
description: Audit a LaTeX manuscript against the Q1 harness and report risks. Use before any submission milestone or when the user asks for a review.
---

# Manuscript Audit

Use when the user wants a full read of where the manuscript stands.

## Procedure
1. `make -f makefiles/ManuscriptHarness.mk harness-verify`.
2. `python scripts/paper_check.py manuscript/main.tex --profile submission --format text`.
3. Group findings by module and translate each into a one-line fix.
4. Rank the top three risks by severity and effort.

## Verification
The audit is complete when every error and warning has a named owner and
a fix. Do not edit during an audit.

## Stop point
Present the report and stop. The user chooses what to fix.
