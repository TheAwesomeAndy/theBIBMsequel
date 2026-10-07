#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan text files for configured phrases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXTENSIONS = {".tex", ".md", ".txt", ".rst"}
EXCLUDES = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def load_phrases(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


# Directories whose content legitimately quotes hype phrases when the
# harness scans itself: rule-violating fixtures, the harness's own rule
# specifications and instructional templates, and generated reports.
SELF_SKIP_DIRS = {"fixtures", "docs", "templates", "playbooks"}
SELF_SKIP_PATHS = {"harness/reports", "harness/memory"}


def iter_files(root: Path, phrase_file: Path | None = None):
    self_scan = root == Path('.')
    phrase_abs = phrase_file.resolve() if phrase_file else None
    for path in root.rglob("*"):
        posix = path.as_posix().lstrip("./")
        if self_scan and path.parts and path.parts[0] in SELF_SKIP_DIRS:
            continue
        if self_scan and any(posix.startswith(skip) for skip in SELF_SKIP_PATHS):
            continue
        # Never scan the phrase-definition file against itself.
        if phrase_abs and path.resolve() == phrase_abs:
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS and not any(part in EXCLUDES for part in path.parts):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--phrase-file", default="harness/forbidden_phrases.txt")
    parser.add_argument("--allow-em-dash", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    phrase_file = Path(args.phrase_file)
    phrases = load_phrases(phrase_file)
    findings = []
    files = [root] if root.is_file() else list(iter_files(root, phrase_file))

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for phrase in phrases:
            if phrase.lower() in lowered:
                findings.append(f"{path}: phrase: {phrase}")
        if not args.allow_em_dash and "—" in text:
            findings.append(f"{path}: em dash")

    if findings:
        print("Phrase check failed:")
        print("\n".join(f"- {item}" for item in findings))
        return 1

    print("Phrase check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
