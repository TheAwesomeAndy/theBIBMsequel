# SPDX-License-Identifier: Apache-2.0
"""Run registered checks against a document and collect findings."""
from __future__ import annotations

import traceback
from pathlib import Path

from papercheck.config import Config
from papercheck.latex import load_document
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import select_checks


def run_checks(
    main_tex: Path,
    config: Config,
    root: Path | None = None,
    select: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[Finding]:
    doc = load_document(main_tex)
    context = CheckContext(doc=doc, config=config, root=root or Path.cwd())
    findings: list[Finding] = []

    for meta in select_checks(select, skip):
        if config.is_disabled(meta.check_id):
            continue
        try:
            results = meta.func(context) or []
        except Exception:  # a broken check must not kill the run
            findings.append(
                Finding(
                    check_id=f"internal.{meta.check_id}",
                    severity=Severity.WARNING,
                    message=f"check crashed: {traceback.format_exc(limit=1).strip()}",
                    path=str(main_tex),
                )
            )
            continue
        for finding in results:
            if config.is_warn_only(finding.check_id) and finding.severity == Severity.ERROR:
                finding.severity = Severity.WARNING
            findings.append(finding)

    findings.sort(key=lambda f: (-f.severity.rank, f.check_id, f.path, f.line))
    return findings


def passed(findings: list[Finding], config: Config) -> bool:
    if any(f.severity == Severity.ERROR for f in findings):
        return False
    if config.strict_warnings and any(f.severity == Severity.WARNING for f in findings):
        return False
    return True
