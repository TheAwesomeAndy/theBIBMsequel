#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare numeric table values in a LaTeX manuscript against CSV data.

This script extracts all numeric values from a LaTeX document and from
one or more CSV files. It then reports mismatches: numbers appearing in
the LaTeX document that are absent from the CSV data, and numbers
appearing in the CSV data that are absent from the LaTeX document.
Use this as a lightweight sanity check that tables in the manuscript
match the underlying result files.

Example:
    python scripts/check_table_values.py manuscript/main.tex outputs/results.csv
    python scripts/check_table_values.py manuscript/main.tex outputs/
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, List

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def extract_numbers_from_latex(tex_path: Path) -> List[float]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    numbers: List[float] = []
    for match in NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(match))
        except ValueError:
            pass
    return numbers


def extract_numbers_from_csv(paths: Iterable[Path]) -> List[float]:
    numbers: List[float] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    for val in row:
                        val_str = val.strip()
                        if not val_str:
                            continue
                        # Remove potential non-numeric characters like commas
                        val_str = val_str.replace(",", "")
                        try:
                            numbers.append(float(val_str))
                        except ValueError:
                            pass
        except Exception:
            # ignore unreadable files
            continue
    return numbers


def collect_csv_files(args_list: list[str]) -> List[Path]:
    csv_files: List[Path] = []
    for arg in args_list:
        p = Path(arg)
        if p.is_dir():
            csv_files.extend(sorted(p.rglob("*.csv")))
        else:
            csv_files.append(p)
    return csv_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latex_file", help="Path to LaTeX file to scan")
    parser.add_argument(
        "csv",
        nargs="+",
        help="CSV files or directories containing CSV files with source values",
    )
    args = parser.parse_args()
    latex_path = Path(args.latex_file)
    if not latex_path.exists():
        print(f"Error: LaTeX file does not exist: {latex_path}", file=sys.stderr)
        return 1
    latex_numbers = extract_numbers_from_latex(latex_path)
    csv_paths = collect_csv_files(list(args.csv))
    csv_numbers = extract_numbers_from_csv(csv_paths)
    missing = [n for n in latex_numbers if n not in csv_numbers]
    extra = [n for n in csv_numbers if n not in latex_numbers]
    if missing or extra:
        print("Mismatch between manuscript tables and CSV data:")
        if missing:
            print(f"Values in LaTeX but not CSV ({len(missing)}): {', '.join(map(str, missing[:20]))}")
            if len(missing) > 20:
                print(f"...and {len(missing) - 20} more")
        if extra:
            print(f"Values in CSV but not LaTeX ({len(extra)}): {', '.join(map(str, extra[:20]))}")
            if len(extra) > 20:
                print(f"...and {len(extra) - 20} more")
        return 1
    print("All numeric table values match CSV data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
