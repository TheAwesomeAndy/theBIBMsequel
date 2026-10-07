#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate manuscript table values against a manifest and source CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Manifest must be JSON-compatible YAML for now: {path}: {exc}"
        ) from exc


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], row_filter: dict[str, Any]) -> dict[str, str] | None:
    for row in rows:
        ok = True
        for key, expected in row_filter.items():
            if str(row.get(key, "")) != str(expected):
                ok = False
                break
        if ok:
            return row
    return None


def as_float(value: Any) -> float:
    return float(str(value).replace(",", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to table manifest")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    failures: list[str] = []

    for table in manifest.get("tables", []):
        table_name = table.get("name", "unnamed_table")
        latex_file = Path(table["latex_file"])
        csv_file = Path(table["csv_file"])

        if not latex_file.exists():
            failures.append(f"{table_name}: missing LaTeX file: {latex_file}")
            continue
        if not csv_file.exists():
            failures.append(f"{table_name}: missing CSV file: {csv_file}")
            continue

        latex_text = latex_file.read_text(encoding="utf-8", errors="replace")
        rows = read_rows(csv_file)

        for item in table.get("matches", []):
            label = item.get("label", "unlabeled_value")
            row = find_row(rows, item.get("csv_row", {}))
            if row is None:
                failures.append(f"{table_name}.{label}: no CSV row matched filter")
                continue

            column = item["csv_column"]
            if column not in row:
                failures.append(f"{table_name}.{label}: missing CSV column {column}")
                continue

            source_value = as_float(row[column])
            manuscript_value = as_float(item["manuscript_value"])
            tolerance = float(item.get("tolerance", 0.0))

            if math.fabs(source_value - manuscript_value) > tolerance:
                failures.append(
                    f"{table_name}.{label}: manuscript {manuscript_value} != CSV {source_value}"
                )
                continue

            required_text = str(item.get("required_text", item["manuscript_value"]))
            if required_text not in latex_text:
                failures.append(
                    f"{table_name}.{label}: required text not found in LaTeX: {required_text}"
                )

    if failures:
        print("table manifest check failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("table manifest check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
