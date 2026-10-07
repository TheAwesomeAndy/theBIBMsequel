#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wall drafting-order gate (Module 3).

A paper is read Head -> Body -> Tail but built like a wall, in order:
figures -> methods -> results -> discussion -> conclusion ->
introduction -> front matter. This gate flags a later phase being
drafted while an earlier one is still thin, so ideas are crystallized
from the concrete core outward.

Usage:
    python scripts/phase_gate.py manuscript/main.tex \\
        --status harness/wall_status.toml --min-words 40
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from papercheck import latex  # noqa: E402

try:
    import tomllib  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

PHASE_ORDER = ["figures", "methods", "results", "discussion", "conclusion", "introduction", "front_matter"]

SECTION_KEYWORDS = {
    "methods": ("method", "methodology", "materials", "experimental setup", "approach"),
    "results": ("result", "findings", "experiments"),
    "discussion": ("discussion",),
    "conclusion": ("conclusion", "concluding"),
    "introduction": ("introduction",),
}


def _section_word_count(sections, keywords) -> int:
    best = 0
    for section in sections:
        if section.title_matches(*keywords):
            best = max(best, len(latex.words(latex.detex(section.body))))
    return best


def phase_status(text: str, min_words: int) -> dict[str, bool]:
    sections = latex.sections(text)
    has_float = bool(latex.environments(text, "figure", "table")) or "\\includegraphics" in text
    status = {"figures": has_float}
    for phase, keywords in SECTION_KEYWORDS.items():
        status[phase] = _section_word_count(sections, keywords) >= min_words
    status["front_matter"] = latex.get_title(text) is not None and latex.get_abstract(text) is not None
    return status


def load_declared_phase(status_path: Path) -> str | None:
    if not status_path.exists() or tomllib is None:
        return None
    with status_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("wall", {}).get("declared_phase")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex", nargs="?", default="manuscript/main.tex")
    parser.add_argument("--status", default="harness/wall_status.toml")
    parser.add_argument("--min-words", type=int, default=40)
    args = parser.parse_args()

    tex_path = Path(args.tex)
    if not tex_path.exists():
        print(f"[ERROR] wall: manuscript not found: {tex_path}")
        return 1

    doc = latex.load_document(tex_path)
    status = phase_status(doc.text, args.min_words)

    for phase in PHASE_ORDER:
        print(f"  {'[done]' if status[phase] else '[todo]'} {phase}")

    violations: list[str] = []
    # A later phase being substantial while an earlier one is thin.
    for i, later in enumerate(PHASE_ORDER):
        if not status[later]:
            continue
        for earlier in PHASE_ORDER[:i]:
            if not status[earlier]:
                violations.append(f"{later} drafted before {earlier}")

    declared = load_declared_phase(Path(args.status))
    if declared and declared in PHASE_ORDER:
        cutoff = PHASE_ORDER.index(declared)
        for earlier in PHASE_ORDER[:cutoff]:
            if not status[earlier]:
                violations.append(f"declared phase {declared} but {earlier} is not done")

    if violations:
        for violation in violations:
            print(f"[ERROR] wall: {violation}")
        return 1
    print("wall check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
