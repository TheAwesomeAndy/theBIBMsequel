---
description: Full pre-submission compliance sweep.
---

Gate the manuscript for submission.

1. Confirm `venue.blinding` and `venue.human_subjects` are set in
   `harness/papercheck.toml`.
2. `python scripts/paper_check.py manuscript/main.tex --profile submission`.
3. If a cover letter exists, `python scripts/check_cover_letter.py <path>`.
4. Verify the data-availability and ethics statements are present.
5. Report a submission checklist with each item marked pass or fail. Any
   fail blocks submission.
