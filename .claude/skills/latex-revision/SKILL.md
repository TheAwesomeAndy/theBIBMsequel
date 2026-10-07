---
name: latex-revision
description: Make small, meaning-preserving edits to a LaTeX manuscript and re-verify. Use for scoped section or sentence revisions.
---

# LaTeX Revision

Use for a narrow edit to one section or passage.

## Procedure
1. State the criteria the revised text must satisfy, including the check
   ids for the section.
2. Make the smallest edit that satisfies them; preserve technical meaning
   and every numeric value.
3. Re-run `python scripts/paper_check.py manuscript/main.tex --profile draft`.

## Verification
The targeted findings clear and no new findings appear. Numbers in the
text still match the source tables (`check_table_manifest.py`).

## Stop point
Show the diff and the before/after finding counts, then stop.
