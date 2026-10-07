# SPDX-License-Identifier: Apache-2.0

MAIN_TEX ?= manuscript/main.tex
PROFILE ?= draft

.PHONY: academic-check paper-check

# Legacy lightweight gate kept for backward compatibility.
academic-check:
	python scripts/qgate.py $(MAIN_TEX)
	python scripts/check_heading.py $(MAIN_TEX)
	python scripts/check_summary.py $(MAIN_TEX)
	python scripts/check_signals.py $(MAIN_TEX)

# The unified engine supersedes the legacy scripts above.
paper-check:
	python scripts/paper_check.py $(MAIN_TEX) --profile $(PROFILE)
