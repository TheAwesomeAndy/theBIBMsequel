# SPDX-License-Identifier: Apache-2.0
"""Render findings as text, markdown, or JSON."""
from __future__ import annotations

import json
from pathlib import Path

from papercheck.config import Config
from papercheck.model import Finding, Severity


def summarize(findings: list[Finding]) -> dict[str, int]:
    return {
        "errors": sum(1 for f in findings if f.severity == Severity.ERROR),
        "warnings": sum(1 for f in findings if f.severity == Severity.WARNING),
        "infos": sum(1 for f in findings if f.severity == Severity.INFO),
    }


def render_text(findings: list[Finding], config: Config, ok: bool) -> str:
    lines: list[str] = []
    counts = summarize(findings)
    if findings:
        for finding in findings:
            lines.append(finding.format_text())
        lines.append("")
    lines.append(
        f"papercheck [{config.profile}]: "
        f"{counts['errors']} error(s), {counts['warnings']} warning(s), {counts['infos']} info(s)"
    )
    lines.append("PASS" if ok else "FAIL")
    return "\n".join(lines)


def render_markdown(findings: list[Finding], config: Config, ok: bool) -> str:
    counts = summarize(findings)
    lines = [
        "# papercheck report",
        "",
        f"- Profile: `{config.profile}`",
        f"- Result: **{'PASS' if ok else 'FAIL'}**",
        f"- Errors: {counts['errors']}, Warnings: {counts['warnings']}, Infos: {counts['infos']}",
        "",
    ]
    if findings:
        lines += [
            "| Severity | Check | Location | Message |",
            "| --- | --- | --- | --- |",
        ]
        for finding in findings:
            location = f"{finding.path}:{finding.line}" if finding.path else ""
            message = finding.message.replace("|", "\\|")
            if finding.suggestion:
                message += f" _(try: {finding.suggestion})_"
            lines.append(
                f"| {finding.severity.value} | `{finding.check_id}` | {location} | {message} |"
            )
    else:
        lines.append("No findings.")
    return "\n".join(lines)


def render_json(findings: list[Finding], config: Config, ok: bool) -> str:
    return json.dumps(
        {
            "profile": config.profile,
            "passed": ok,
            "summary": summarize(findings),
            "findings": [finding.to_dict() for finding in findings],
        },
        indent=2,
    )


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
