#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Check a LaTeX log file for undefined citations and references.

This utility scans LaTeX log files produced by pdflatex/latexmk and
reports warnings about undefined citations and references. It returns
non-zero if any such problems are found.

Example:
    # after compiling a document
    python scripts/check_citations.py manuscript/main.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_log(log_text: str) -> list[str]:
    """Extract warning lines for undefined citations/references."""
    findings: list[str] = []
    for line in log_text.splitlines():
        # Catch typical LaTeX warnings about undefined citations or references
        if re.search(r"(Citation|Reference).*undefined", line, re.IGNORECASE):
            findings.append(line.strip())
        if "undefined citations" in line.lower() or "undefined references" in line.lower():
            findings.append(line.strip())
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        help="Path to LaTeX log (.log) file produced by compiling the manuscript.",
    )
    args = parser.parse_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: log file does not exist: {log_path}", file=sys.stderr)
        return 1
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    problems = parse_log(text)
    if problems:
        print("Undefined citations/references found:")
        for item in problems:
            print(f"- {item}")
        return 1
    print("No undefined citations or references detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
