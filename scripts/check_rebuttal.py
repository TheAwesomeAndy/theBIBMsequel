#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuttal gate (Module 5): the bullfighter's point-by-point reply.

Enforces the winner's mindset: every reviewer comment gets a polite,
itemized response with an exact page/line/section locator, and no
emotional language survives the 7-day cooling-off.

Usage:
    python scripts/check_rebuttal.py submission/rebuttal.md [--strict]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papercheck import vocab  # noqa: E402

COMMENT_RE = re.compile(
    r"^\s*(?:#+\s*|\*+\s*)?(?:reviewer\s*\d+|r\d+(?:\.\d+)?|comment\s*\d+|point\s*\d+|q\d+|concern\s*\d+)\b[:.]?",
    re.IGNORECASE,
)
RESPONSE_RE = re.compile(
    r"^\s*(?:#+\s*|\*+\s*)?(?:response|reply|answer|our response)\b[:.]?",
    re.IGNORECASE,
)
LOCATOR_RE = re.compile(
    r"\b(?:page|p\.|line|l\.|lines|section|sec\.)\s*~?\s*\d+",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rebuttal", help="Path to the rebuttal (.md)")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings too")
    args = parser.parse_args()

    path = Path(args.rebuttal)
    if not path.exists():
        print(f"[ERROR] rebuttal: file not found: {path}")
        return 1

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lowered = "\n".join(lines).lower()
    errors: list[str] = []
    warnings: list[str] = []

    # Segment into comment blocks; each must contain a response.
    comment_indices = [i for i, line in enumerate(lines) if COMMENT_RE.match(line)]
    if not comment_indices:
        errors.append("no reviewer/comment markers found; the rebuttal is not point-by-point")
    else:
        bounds = comment_indices + [len(lines)]
        for n, start in enumerate(comment_indices):
            block = lines[start : bounds[n + 1]]
            block_text = "\n".join(block)
            if not any(RESPONSE_RE.match(line) for line in block[1:]):
                label = lines[start].strip()[:50]
                errors.append(f"comment {label!r} has no response")
            elif not LOCATOR_RE.search(block_text):
                label = lines[start].strip()[:50]
                warnings.append(f"response to {label!r} has no page/line/section locator")

    for phrase in vocab.EMOTIONAL_REBUTTAL:
        if phrase in lowered:
            errors.append(f"emotional phrasing {phrase!r}; strip emotion, stay collegial")

    if not any(marker in lowered for marker in vocab.POLITENESS_MARKERS):
        warnings.append("no courtesy language found; open each response by thanking the reviewer")

    for finding in errors:
        print(f"[ERROR] rebuttal: {finding}")
    for finding in warnings:
        print(f"[WARNING] rebuttal: {finding}")

    if errors or (args.strict and warnings):
        return 1
    print("rebuttal check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
