#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compile a LaTeX document using latexmk or pdflatex and report errors.

This script attempts to build a LaTeX project in PDF mode. It uses
`latexmk` if available, falling back to `pdflatex`. By default the tool
prints the build log and returns non-zero when compilation fails.

Example:
    python scripts/verify_latex.py manuscript/main.tex
"""
from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


def build_latex(tex_file: str) -> int:
    # The build runs inside the file's own directory (so relative \input and
    # \includegraphics paths resolve), so pass the bare file name: passing the
    # caller-relative path from inside that directory fails for any manuscript
    # that lives in a subdirectory (e.g. manuscript/main.tex).
    tex_name = Path(tex_file).name
    # Prefer latexmk because it handles multi-pass compilation automatically.
    latexmk = shutil.which("latexmk")
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", tex_name]
    else:
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            cmd = [pdflatex, "-interaction=nonstopmode", tex_name]
        else:
            print("Error: neither latexmk nor pdflatex is installed.", file=sys.stderr)
            return 1
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(tex_file).parent),
    )
    print(proc.stdout)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tex_file",
        help="Path to the root .tex file to compile (e.g. manuscript/main.tex)",
    )
    args = parser.parse_args()
    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: file not found: {tex_path}", file=sys.stderr)
        return 1
    ret = build_latex(str(tex_path))
    if ret != 0:
        print("LaTeX compilation failed.")
        return 1
    print("LaTeX compilation succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
