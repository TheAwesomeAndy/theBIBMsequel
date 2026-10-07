# SPDX-License-Identifier: Apache-2.0
"""Command-line interface: ``python -m papercheck manuscript/main.tex``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from papercheck.config import PROFILES, Config
from papercheck.registry import all_checks
from papercheck.report import render_json, render_markdown, render_text, write_report
from papercheck.runner import passed, run_checks


def _split_patterns(values: list[str]) -> list[str]:
    patterns: list[str] = []
    for value in values or []:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papercheck",
        description="Deterministic Q1-publication quality gates for LaTeX manuscripts.",
    )
    parser.add_argument("tex", nargs="?", help="Main .tex file (default: paper.main_tex from config)")
    parser.add_argument("--config", default=None, help="Path to papercheck.toml (default: harness/papercheck.toml)")
    parser.add_argument("--profile", choices=PROFILES, default="draft", help="Strictness profile")
    parser.add_argument("--select", action="append", default=[], help="Only run these check ids/prefixes (comma separated, repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="Skip these check ids/prefixes")
    parser.add_argument("--format", choices=("text", "json", "md"), default="text", help="Output format")
    parser.add_argument("--report", default=None, help="Also write a JSON report to this path")
    parser.add_argument(
        "--history",
        nargs="?",
        const="harness/reports/history.jsonl",
        default=None,
        help="Append a one-line JSON run summary to this file (default: harness/reports/history.jsonl)",
    )
    parser.add_argument("--list-checks", action="store_true", help="List registered checks and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_checks:
        for meta in sorted(all_checks().values(), key=lambda m: m.check_id):
            print(f"{meta.check_id:32s} [{meta.module}] {meta.title}")
        return 0

    config_path = Path(args.config) if args.config else Path("harness/papercheck.toml")
    config = Config.load(config_path if config_path.exists() else None, profile=args.profile)

    tex = args.tex or config.get("paper.main_tex")
    tex_path = Path(tex)
    if not tex_path.exists():
        print(f"error: manuscript not found: {tex_path}", file=sys.stderr)
        return 2

    findings = run_checks(
        tex_path,
        config,
        root=Path.cwd(),
        select=_split_patterns(args.select),
        skip=_split_patterns(args.skip),
    )
    ok = passed(findings, config)

    if args.format == "json":
        output = render_json(findings, config, ok)
    elif args.format == "md":
        output = render_markdown(findings, config, ok)
    else:
        output = render_text(findings, config, ok)
    print(output)

    if args.report:
        write_report(Path(args.report), render_json(findings, config, ok))

    if args.history:
        append_history(Path(args.history), tex_path, config, findings, ok)

    return 0 if ok else 1


def append_history(path: Path, tex_path: Path, config: Config, findings, ok: bool) -> None:
    """Append a one-line JSON run summary (Module 7: run history)."""
    import json
    import time

    from papercheck.report import summarize

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tex": str(tex_path),
        "profile": config.profile,
        "passed": ok,
        "summary": summarize(findings),
        "check_ids": sorted({finding.check_id for finding in findings}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
