#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-improvement analyzer (Module 7, Rules 7.3-7.5).

Mines the papercheck run history and the project memory journal for
recurring findings, quality trends, chronic failures, stale memory, and
reviewer echoes, then emits concrete, human-approvable harness upgrades.
The harness learns from its own runs: a mistake that recurs becomes a
rule.

Usage:
    python scripts/self_improve.py --last 20 --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_HISTORY = "harness/reports/history.jsonl"
DEFAULT_MEMORY = "harness/memory/memory.jsonl"
DEFAULT_REPORT = "harness/reports/improvement_report.md"

# family prefix -> concrete upgrade action
UPGRADE_ACTIONS = {
    "lang": "add the offending phrases to harness/forbidden_phrases.txt and tighten the never-do list in CLAUDE.md",
    "claims": "add the offending phrases to harness/forbidden_phrases.txt and tighten the never-do list in CLAUDE.md",
    "intro": "add the rule to the drafting checklist in playbooks/sections.md and re-run /draft-phase",
    "structure": "add the rule to the drafting checklist in playbooks/sections.md and re-run /draft-phase",
    "limits": "add the rule to the drafting checklist in playbooks/sections.md and re-run /draft-phase",
    "methods": "record a decision entry documenting the rigor fix and update harness/papercheck.toml (venue.human_subjects, checks.warn_only)",
    "bib": "fix the .bib hygiene and add a pre-commit 'make harness-paper' run",
    "blind": "update harness/papercheck.toml venue settings and add the scrub step to the submission checklist",
    "ethics": "update harness/papercheck.toml venue settings and add the scrub step to the submission checklist",
    "data": "update harness/papercheck.toml venue settings and add the scrub step to the submission checklist",
    "floats": "regenerate figures with grayscale-safe styles (varied strokes and markers)",
}


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _family(check_id: str) -> str:
    return check_id.split(".", 1)[0]


def analyze(history: list[dict], memory: list[dict]):
    result: dict = {
        "recurring": [],
        "trend": "flat",
        "chronic": [],
        "stale": [],
        "reviewer_echoes": [],
    }
    if history:
        # Recurring findings: check ids in >= 3 of the runs.
        counts: dict[str, int] = {}
        for run in history:
            for check_id in set(run.get("check_ids", [])):
                counts[check_id] = counts.get(check_id, 0) + 1
        result["recurring"] = sorted(
            [(cid, n) for cid, n in counts.items() if n >= 3],
            key=lambda pair: (-pair[1], pair[0]),
        )

        # Chronic families: family present in every run.
        run_families = [set(_family(c) for c in run.get("check_ids", [])) for run in history]
        if run_families:
            chronic = set.intersection(*run_families) if all(run_families) else set()
            result["chronic"] = sorted(chronic)

        # Trend: sum of errors+warnings, older half vs newer half.
        def load(run: dict) -> int:
            summary = run.get("summary", {})
            return summary.get("errors", 0) + summary.get("warnings", 0)

        half = len(history) // 2
        if half:
            older = sum(load(run) for run in history[:half]) / half
            newer = sum(load(run) for run in history[half:]) / (len(history) - half)
            if newer < older - 0.5:
                result["trend"] = "improving"
            elif newer > older + 0.5:
                result["trend"] = "regressing"

    # Stale memory: open entries older than the window, or open questions.
    oldest_ts = history[0].get("timestamp", "") if history else ""
    for entry in memory:
        if entry.get("status") != "open":
            continue
        if entry.get("kind") == "question" and not history:
            result["stale"].append(entry)
        elif oldest_ts and entry.get("timestamp", "") < oldest_ts:
            result["stale"].append(entry)

    # Reviewer echoes: open reviewer entries whose words match a recurring family.
    recurring_families = {_family(cid) for cid, _ in result["recurring"]}
    for entry in memory:
        if entry.get("status") != "open" or entry.get("kind") != "reviewer":
            continue
        words = {w.lower() for w in entry.get("text", "").split() if len(w) >= 6}
        if words & {fam.lower() for fam in recurring_families}:
            result["reviewer_echoes"].append(entry)
    return result


def render(result: dict) -> str:
    lines = ["# Self-improvement report", ""]
    lines.append(f"## Trend\n\n{result['trend']}\n")

    lines.append("## Recurring findings\n")
    if result["recurring"]:
        lines.append("| check id | runs |")
        lines.append("| --- | --- |")
        for cid, n in result["recurring"]:
            lines.append(f"| `{cid}` | {n} |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Chronic families\n")
    lines.append(", ".join(f"`{fam}`" for fam in result["chronic"]) if result["chronic"] else "None.")
    lines.append("")

    lines.append("## Stale memory\n")
    if result["stale"]:
        for entry in result["stale"]:
            lines.append(f"- #{entry.get('id')} [{entry.get('kind')}] {entry.get('text')}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Reviewer echoes\n")
    if result["reviewer_echoes"]:
        for entry in result["reviewer_echoes"]:
            lines.append(f"- #{entry.get('id')} {entry.get('text')} (a reviewer already flagged this; encode it as a rule)")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## Suggested harness upgrades\n")
    families = {_family(cid) for cid, _ in result["recurring"]} | set(result["chronic"])
    actions = []
    for family in sorted(families):
        if family in UPGRADE_ACTIONS:
            actions.append(f"- `{family}.*`: {UPGRADE_ACTIONS[family]}")
    if result["reviewer_echoes"]:
        actions.append("- close the loop: turn each reviewer echo above into a deterministic check or standing rule (Rule 7.5)")
    if result["stale"]:
        actions.append("- resolve or re-scope the stale memory entries above with `scripts/memory.py resolve`")
    lines.extend(actions if actions else ["None. The harness is keeping up."])
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=DEFAULT_HISTORY)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    history = _load_jsonl(Path(args.history))[-args.last :]
    memory = _load_jsonl(Path(args.memory))

    if not history and not memory:
        print("no history or memory yet; run papercheck with --history and record memory entries")
        return 0

    result = analyze(history, memory)
    report = render(result)
    print(report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")

    if args.strict and (result["trend"] == "regressing" or result["chronic"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
