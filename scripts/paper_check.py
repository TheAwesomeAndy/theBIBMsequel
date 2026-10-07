#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper so the unified checker follows the scripts/ convention.

Equivalent to ``python -m papercheck``. Example:

    python scripts/paper_check.py manuscript/main.tex --profile submission
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papercheck.cli import main  # noqa: E402

raise SystemExit(main())
