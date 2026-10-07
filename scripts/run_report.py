#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run core checks and write JSON."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

commands = [
    ["python", "scripts/check_forbidden_phrases.py", ".", "--phrase-file", "harness/forbidden_phrases.txt"],
    ["python", "tools/check_path_markers.py", "."],
]

results = []
for command in commands:
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results.append({"command": command, "returncode": proc.returncode, "passed": proc.returncode == 0})

report = {"results": results, "passed": all(item["passed"] for item in results)}
path = Path("harness/reports/run_report.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote {path}")
raise SystemExit(0 if report["passed"] else 1)
