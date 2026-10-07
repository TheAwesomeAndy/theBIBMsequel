#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Project memory journal (Module 7, Rule 7.2).

Durable, inspectable knowledge that survives context loss: decisions
(with rationale), lessons (what a gate caught and why), reviewer
feedback (verbatim), and open questions. Stored as one JSON object per
line in harness/memory/memory.jsonl.

Subcommands:
    add <kind> <text> [--tags a,b]   append an entry, print its id
    list [--kind K] [--status S] [--tag T]
    search <term>
    resolve <id> <resolution>
    digest [--limit N]               compact block for an AI context window
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_FILE = "harness/memory/memory.jsonl"
KINDS = ("decision", "lesson", "reviewer", "question")
STATUSES = ("open", "resolved")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: skipping malformed line {number} in {path}", file=sys.stderr)
    return entries


def save(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _format(entry: dict) -> str:
    tags = ", ".join(entry.get("tags", []))
    tag_str = f" (tags: {tags})" if tags else ""
    line = (
        f"#{entry.get('id')} [{entry.get('kind')}/{entry.get('status')}] "
        f"{entry.get('timestamp', '')} {entry.get('text', '')}{tag_str}"
    )
    if entry.get("status") == "resolved" and entry.get("resolution"):
        line += f"\n    -> {entry['resolution']}"
    return line


def cmd_add(args) -> int:
    if args.kind not in KINDS:
        print(f"error: kind must be one of {KINDS}", file=sys.stderr)
        return 1
    path = Path(args.file)
    entries = load(path)
    next_id = max((entry.get("id", 0) for entry in entries), default=0) + 1
    tags = [tag.strip() for tag in (args.tags or "").split(",") if tag.strip()]
    entries.append(
        {
            "id": next_id,
            "timestamp": _timestamp(),
            "kind": args.kind,
            "text": args.text,
            "tags": tags,
            "status": "open",
        }
    )
    save(path, entries)
    print(next_id)
    return 0


def cmd_list(args) -> int:
    entries = load(Path(args.file))
    for entry in entries:
        if args.kind and entry.get("kind") != args.kind:
            continue
        if args.status and entry.get("status") != args.status:
            continue
        if args.tag and args.tag not in entry.get("tags", []):
            continue
        print(_format(entry))
    return 0


def cmd_search(args) -> int:
    term = args.term.lower()
    for entry in load(Path(args.file)):
        haystack = (entry.get("text", "") + " " + " ".join(entry.get("tags", []))).lower()
        if term in haystack:
            print(_format(entry))
    return 0


def cmd_resolve(args) -> int:
    path = Path(args.file)
    entries = load(path)
    for entry in entries:
        if entry.get("id") == args.id:
            entry["status"] = "resolved"
            entry["resolution"] = args.resolution
            save(path, entries)
            print(f"resolved #{args.id}")
            return 0
    print(f"error: no entry with id {args.id}", file=sys.stderr)
    return 1


def cmd_digest(args) -> int:
    entries = load(Path(args.file))
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for entry in entries:
        by_kind[entry.get("kind", "?")] = by_kind.get(entry.get("kind", "?"), 0) + 1
        by_status[entry.get("status", "?")] = by_status.get(entry.get("status", "?"), 0) + 1

    print("=== project memory digest ===")
    print("counts by kind: " + (", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "none"))
    print("counts by status: " + (", ".join(f"{k}={v}" for k, v in sorted(by_status.items())) or "none"))
    open_entries = [entry for entry in entries if entry.get("status") == "open"]
    open_entries.sort(key=lambda entry: entry.get("id", 0), reverse=True)
    print(f"\nmost recent open entries (up to {args.limit}):")
    if not open_entries:
        print("  (none)")
    for entry in open_entries[: args.limit]:
        print("  " + _format(entry))
    print(f"\nFull journal: {args.file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="append an entry")
    p_add.add_argument("kind", choices=KINDS)
    p_add.add_argument("text")
    p_add.add_argument("--tags", default="")
    p_add.add_argument("--file", default=DEFAULT_FILE)
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="list entries")
    p_list.add_argument("--kind", choices=KINDS)
    p_list.add_argument("--status", choices=STATUSES)
    p_list.add_argument("--tag")
    p_list.add_argument("--file", default=DEFAULT_FILE)
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="substring search")
    p_search.add_argument("term")
    p_search.add_argument("--file", default=DEFAULT_FILE)
    p_search.set_defaults(func=cmd_search)

    p_resolve = sub.add_parser("resolve", help="mark an entry resolved")
    p_resolve.add_argument("id", type=int)
    p_resolve.add_argument("resolution")
    p_resolve.add_argument("--file", default=DEFAULT_FILE)
    p_resolve.set_defaults(func=cmd_resolve)

    p_digest = sub.add_parser("digest", help="compact context block")
    p_digest.add_argument("--limit", type=int, default=20)
    p_digest.add_argument("--file", default=DEFAULT_FILE)
    p_digest.set_defaults(func=cmd_digest)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
