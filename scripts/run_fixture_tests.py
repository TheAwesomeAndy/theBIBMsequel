#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run synthetic fixture tests for the research AI harness.

Two kinds of cases run here:

1. Hard-coded legacy cases exercising the standalone scripts.
2. Discovered papercheck cases: every ``fixtures/paperlint/*/case.json``
   declares a manuscript plus the check ids it must pass or fail:

       {
         "name": "intro_roadmap",
         "tex": "manuscript/main.tex",
         "select": ["intro.roadmap"],
         "profile": "submission",
         "expect": "fail",
         "config": "harness/papercheck.toml"   // optional, case-relative
       }
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureCase:
    name: str
    command: list[str]
    expect_success: bool


def discover_paperlint_cases(python: str, root: Path = Path("fixtures/paperlint")) -> list[FixtureCase]:
    cases: list[FixtureCase] = []
    if not root.exists():
        return cases
    for case_file in sorted(root.glob("*/case.json")):
        spec = json.loads(case_file.read_text(encoding="utf-8"))
        case_dir = case_file.parent
        command = [
            python,
            "-m",
            "papercheck",
            str(case_dir / spec.get("tex", "manuscript/main.tex")),
            "--profile",
            spec.get("profile", "submission"),
        ]
        select = spec.get("select", [])
        if select:
            command += ["--select", ",".join(select)]
        for pattern in spec.get("skip", []):
            command += ["--skip", pattern]
        config = spec.get("config")
        if config:
            command += ["--config", str(case_dir / config)]
        cases.append(
            FixtureCase(
                name=spec.get("name", case_dir.name),
                command=command,
                expect_success=spec.get("expect", "fail") == "pass",
            )
        )
    return cases


def run_case(case: FixtureCase) -> dict[str, object]:
    proc = subprocess.run(case.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    success = proc.returncode == 0
    passed = success == case.expect_success
    print(f"[{'PASS' if passed else 'FAIL'}] {case.name}")
    if not passed:
        print(proc.stdout)
    return {
        "name": case.name,
        "expected_success": case.expect_success,
        "actual_success": success,
        "passed": passed,
        "returncode": proc.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="harness/reports/fixture_report.json")
    args = parser.parse_args()
    python = sys.executable
    has_latex = shutil.which("latexmk") or shutil.which("pdflatex")

    cases = [
        FixtureCase("good_manifest", [python, "scripts/check_table_manifest.py", "fixtures/good_manuscript/harness/table_manifest.yaml"], True),
        FixtureCase("bad_table_value", [python, "scripts/check_table_manifest.py", "fixtures/bad_table_value/harness/table_manifest.yaml"], False),
        FixtureCase("bad_citation_log", [python, "scripts/check_citations.py", "fixtures/bad_citation/manuscript/main.log"], False),
        FixtureCase("forbidden_phrase", [python, "scripts/check_forbidden_phrases.py", "fixtures/forbidden_phrase", "--phrase-file", "harness/forbidden_phrases.txt"], False),
        FixtureCase("path_marker", [python, "tools/check_path_markers.py", "fixtures/path_marker"], False),
        # Submission gate scripts.
        FixtureCase("cover_letter_good", [python, "scripts/check_cover_letter.py", "fixtures/submission/cover_letter_good.md", "--journal", "Journal of Robust Machine Intelligence"], True),
        FixtureCase("cover_letter_bad", [python, "scripts/check_cover_letter.py", "fixtures/submission/cover_letter_bad.md"], False),
        FixtureCase("rebuttal_good", [python, "scripts/check_rebuttal.py", "fixtures/submission/rebuttal_good.md"], True),
        FixtureCase("rebuttal_bad", [python, "scripts/check_rebuttal.py", "fixtures/submission/rebuttal_bad.md"], False),
        FixtureCase("wall_good", [python, "scripts/phase_gate.py", "fixtures/wall/good/manuscript/main.tex"], True),
        FixtureCase("wall_violation", [python, "scripts/phase_gate.py", "fixtures/wall/violation/manuscript/main.tex"], False),
        # Memory tooling smoke checks (Module 7).
        FixtureCase("memory_digest", [python, "scripts/memory.py", "digest", "--file", "fixtures/memory/memory.jsonl"], True),
        FixtureCase("self_improve", [python, "scripts/self_improve.py", "--history", "fixtures/memory/history.jsonl", "--memory", "fixtures/memory/memory.jsonl", "--report", "harness/reports/improvement_report.md"], True),
    ]

    if has_latex:
        cases.append(FixtureCase("good_latex_build", [python, "scripts/verify_latex.py", "fixtures/good_manuscript/manuscript/main.tex"], True))
        cases.append(FixtureCase("bad_latex_build", [python, "scripts/verify_latex.py", "fixtures/bad_latex/manuscript/main.tex"], False))
    else:
        print("[SKIP] LaTeX build fixtures require latexmk or pdflatex")

    cases.extend(discover_paperlint_cases(python))

    results = [run_case(case) for case in cases]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    if any(not item["passed"] for item in results):
        print("fixture suite failed")
        return 1
    print("fixture suite passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
