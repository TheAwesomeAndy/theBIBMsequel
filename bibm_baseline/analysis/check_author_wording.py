#!/usr/bin/env python3
"""Enforce the author's standing wording rules on the paper (memory journal entry #16).

Banned in the title and in the prose (abstract through conclusion, including figure and
table captions; the bibliography is not scanned): 'observable', 'characterization' /
'characterize', 'audit', 'behavior', 'response', 'metric'; 'robustness' is banned in the
title only. The harness's repo-wide phrase gate (scripts/check_forbidden_phrases.py) is not
used for these because it also scans reference titles, code, and documentation, where these
common words are legitimate.

Usage: python analysis/check_author_wording.py [manuscript/main_bibm2026.tex]
Exit status 1 if any banned word is present.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED_PROSE = ["observable", "characteriz", "audit", "behavior", "behaviour", "response", "metric"]
BANNED_TITLE_ONLY = ["robustness"]


def main() -> int:
    tex = Path(sys.argv[1] if len(sys.argv) > 1 else "manuscript/main_bibm2026.tex")
    s = tex.read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]*)\}", s).group(1)
    start = s.index("\\begin{abstract}")
    end = s.index("\\begin{thebibliography}")
    body = s[start:end]
    first_line = s[:start].count("\n") + 1
    findings = []
    for w in BANNED_PROSE + BANNED_TITLE_ONLY:
        if re.search(w, title, re.I):
            findings.append(f"title: '{w}' in \"{title}\"")
    for i, line in enumerate(body.split("\n")):
        stripped = re.sub(r"(?<!\\)%.*$", "", line)                 # ignore LaTeX comments
        for w in BANNED_PROSE:
            for m in re.finditer(w, stripped, re.I):
                ctx = stripped[max(0, m.start() - 40):m.end() + 30]
                findings.append(f"{tex}:{first_line + i}: '{w}' ...{ctx}...")
    if findings:
        print("Author wording check failed:")
        print("\n".join(f"- {f}" for f in findings))
        return 1
    print("Author wording check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
