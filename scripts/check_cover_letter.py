#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cover-letter gate (Module 5): the sober three-paragraph pitch.

A Q1 cover letter is a high-stakes, un-hyped sales pitch of at most one
page: (1) what the paper says, (2) why it matters, (3) why it fits this
journal's ongoing conversation.

Usage:
    python scripts/check_cover_letter.py submission/cover_letter.md \\
        --journal "Nature Machine Intelligence"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papercheck import vocab  # noqa: E402
from papercheck.config import Config  # noqa: E402

MIN_PARAGRAPH_WORDS = 15


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z\-']*", text))


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("letter", help="Path to the cover letter (.md or .txt)")
    parser.add_argument("--journal", default=None)
    parser.add_argument("--config", default="harness/papercheck.toml")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--paragraphs", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.letter)
    if not path.exists():
        print(f"[ERROR] cover-letter: file not found: {path}")
        return 1

    config_path = Path(args.config)
    config = Config.load(config_path if config_path.exists() else None)
    max_words = args.max_words if args.max_words is not None else config.get("cover_letter.max_words", 400)
    want_paragraphs = args.paragraphs if args.paragraphs is not None else config.get("cover_letter.paragraphs", 3)
    journal = args.journal or config.get("venue.name", "")

    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    findings: list[str] = []

    blocks = _paragraphs(text)
    substantive = [block for block in blocks if _words(block) >= MIN_PARAGRAPH_WORDS]
    if len(substantive) != want_paragraphs:
        findings.append(
            f"expected {want_paragraphs} substantive paragraphs, found {len(substantive)} "
            "(what it says / why it matters / why it fits)"
        )

    total_words = _words(text)
    if total_words > max_words:
        findings.append(f"letter is {total_words} words; keep it under {max_words} (one page)")

    for phrase in list(vocab.HYPE_ERROR) + list(vocab.OVERCLAIM):
        if phrase in lowered:
            findings.append(f"hype/overclaim phrase {phrase!r}; keep the pitch sober")

    if journal and journal.lower() not in lowered:
        findings.append(f"journal name {journal!r} is not mentioned; tailor the fit paragraph")

    if findings:
        for finding in findings:
            print(f"[ERROR] cover-letter: {finding}")
        return 1
    print("cover letter check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
