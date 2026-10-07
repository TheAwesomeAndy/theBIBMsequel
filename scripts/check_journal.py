#!/usr/bin/env python3
"""Journal manuscript gate: regenerate values, build, and check the result.

Usage: python scripts/check_journal.py [--no-export]
Steps (all must pass):
  1. analysis/export_journal_values.py (unless --no-export);
  2. latexmk build of manuscript/journal/main_journal.tex;
  3. log: no LaTeX errors, undefined references or citations, missing figures, or
     overfull boxes;
  4. no draft markers (PLACEHOLDER, RESULT-, DISC-, DRAFT, IfFileExists) in the source;
  5. author wording rules (analysis/check_author_wording.py) plus the style rules used
     for this manuscript: no em dashes or semicolons in the prose, none of the words
     robust, delve, leverage, tapestry, unlock, revolutionary, foster, testament;
  6. bibliography: every cited key has an item and every item is cited;
  7. abstract at most 250 words.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "manuscript" / "journal" / "main_journal.tex"
BANNED = ["robust", "delve", "leverage", "tapestry", "unlock", "revolutionary", "foster", "testament"]


def run(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    fails = []
    if "--no-export" not in sys.argv:
        r = run([sys.executable, "analysis/export_journal_values.py"])
        print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
        if r.returncode:
            fails.append("export failed")
    r = run(["latexmk", "-pdf", "-interaction=nonstopmode", "main_journal.tex"], cwd=TEX.parent)
    log = (TEX.parent / "main_journal.log").read_text(errors="ignore")
    for pat, name in ((r"^! ", "LaTeX error"), (r"undefined", "undefined reference/citation"),
                      (r"Overfull", "overfull box"), (r"File `[^']*' not found", "missing file")):
        n = len(re.findall(pat, log, re.M))
        if n:
            fails.append(f"{n} x {name}")
    pages = re.search(r"Output written on .*\((\d+) pages", log)
    print(f"[journal] build: {pages.group(1) if pages else '?'} pages")
    s = TEX.read_text(encoding="utf-8")
    for m in ("PLACEHOLDER", "RESULT-", "DISC-", "DRAFT", "IfFileExists"):
        if m in s:
            fails.append(f"draft marker '{m}' present")
    r = run([sys.executable, "analysis/check_author_wording.py", str(TEX.relative_to(ROOT))])
    if r.returncode:
        fails.append("author wording: " + r.stdout.strip().replace("\n", " | "))
    body = s[s.index("\\begin{abstract}"):s.index("\\begin{thebibliography}")]
    body = re.sub(r"(?<!\\)%.*", "", body)
    if ";" in body:
        fails.append(f"{body.count(';')} semicolon(s) in the prose")
    if "—" in body or "---" in body:
        fails.append("em dash in the prose")
    for w in BANNED:
        if re.search(w, body, re.I):
            fails.append(f"banned word '{w}'")
    r = run([sys.executable, "scripts/order_bibliography.py", str(TEX.relative_to(ROOT))])
    print(r.stdout.strip())
    if "missing []" not in r.stdout or "uncited []" not in r.stdout:
        fails.append("bibliography coverage")
    a = s[s.index("\\begin{abstract}") + 16:s.index("\\end{abstract}")]
    n_words = len(re.sub(r"\$[^$]*\$", "X", a).split())
    print(f"[journal] abstract: {n_words} words")
    if n_words > 250:
        fails.append(f"abstract {n_words} words")
    for f in fails:
        print("[journal] FAIL: " + f)
    print("[journal] " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
