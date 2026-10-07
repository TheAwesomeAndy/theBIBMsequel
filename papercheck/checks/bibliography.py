# SPDX-License-Identifier: Apache-2.0
"""Module BIB: BibTeX hygiene.

A small stdlib BibTeX reader discovers .bib files from the config, from
\\bibliography{...}, and from \\addbibresource{...}, then checks that
every cited key resolves, keys are unique, entries carry the core
fields, no entries are dead, and the literature is not stale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from papercheck import latex
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
FIELD_RE = re.compile(r"(\w+)\s*=", re.IGNORECASE)
YEAR_RE = re.compile(r"(\d{4})")
SKIP_TYPES = {"string", "preamble", "comment"}


@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: set[str]
    year: int | None
    source: str


@dataclass
class BibData:
    entries: list[BibEntry] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    def keys(self) -> set[str]:
        return {entry.key for entry in self.entries}


def _skip_group(text: str, index: int) -> int:
    """Skip a balanced {..} or "..." field value starting at index."""
    if index >= len(text):
        return index
    char = text[index]
    if char == "{":
        depth = 0
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        return index
    if char == '"':
        index += 1
        while index < len(text) and text[index] != '"':
            index += 1
        return index + 1
    return index


def parse_bib(text: str, source: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    for match in ENTRY_RE.finditer(text):
        entry_type = match.group(1).lower()
        if entry_type in SKIP_TYPES:
            continue
        key = match.group(2).strip()
        # Bound the entry body by the next @entry or EOF.
        next_at = text.find("@", match.end())
        body = text[match.end() : next_at if next_at >= 0 else len(text)]
        fields: set[str] = set()
        year: int | None = None
        pos = 0
        while pos < len(body):
            field_match = FIELD_RE.match(body, pos)
            if not field_match:
                pos += 1
                continue
            name = field_match.group(1).lower()
            fields.add(name)
            value_start = field_match.end()
            while value_start < len(body) and body[value_start] in " \t\n":
                value_start += 1
            value_end = _skip_group(body, value_start)
            if name in ("year", "date") and year is None:
                year_match = YEAR_RE.search(body[value_start:value_end])
                if year_match:
                    year = int(year_match.group(1))
            pos = max(value_end, value_start + 1)
        entries.append(
            BibEntry(key=key, entry_type=entry_type, fields=fields, year=year, source=source)
        )
    return entries


def discover_bib(ctx: CheckContext) -> BibData:
    main_dir = ctx.doc.main_path.parent
    candidates: list[Path] = []

    def add(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if not name.endswith(".bib"):
            name = name + ".bib"
        for base in (main_dir, Path.cwd()):
            path = (base / name).resolve()
            if path.exists() and path not in candidates:
                candidates.append(path)
                return

    for entry in ctx.config.get("paper.bib", []) or []:
        add(entry)
    for match in re.finditer(r"\\bibliography\s*\{([^{}]+)\}", ctx.doc.text):
        for name in match.group(1).split(","):
            add(name)
    for match in re.finditer(r"\\addbibresource\s*\{([^{}]+)\}", ctx.doc.text):
        add(match.group(1))

    data = BibData(files=candidates)
    for path in candidates:
        data.entries.extend(parse_bib(path.read_text(encoding="utf-8", errors="replace"), str(path)))
    return data


@register("bib.files", "Bibliography files are discoverable", "BIB")
def bib_files(ctx: CheckContext) -> list[Finding]:
    """If the paper cites but no .bib is found, reference checks cannot run."""
    cited = latex.cite_keys(ctx.doc.text)
    if not cited:
        return []
    if discover_bib(ctx).files:
        return []
    return [
        ctx.finding(
            "bib.files",
            Severity.INFO,
            "manuscript has citations but no .bib file was found; reference checks skipped",
            pos=cited[0][1],
            rule="BIB.files",
            suggestion="add \\bibliography{...} or set paper.bib in harness/papercheck.toml",
        )
    ]


@register("bib.missing-entries", "Every citation resolves to an entry", "BIB")
def bib_missing_entries(ctx: CheckContext) -> list[Finding]:
    """A citation with no matching .bib entry compiles to a dangling '?'."""
    data = discover_bib(ctx)
    if not data.files:
        return []
    keys = data.keys()
    findings: list[Finding] = []
    seen: set[str] = set()
    for key, pos in latex.cite_keys(ctx.doc.text):
        if "*" in key or key in seen:
            continue
        seen.add(key)
        if key not in keys:
            findings.append(
                ctx.finding(
                    "bib.missing-entries",
                    Severity.ERROR,
                    f"cited key {key!r} is not defined in any bib file",
                    pos=pos,
                    rule="BIB.missing-entries",
                    suggestion="add the entry or fix the citation key",
                )
            )
    return findings


@register("bib.duplicate-keys", "Bib keys are unique", "BIB")
def bib_duplicate_keys(ctx: CheckContext) -> list[Finding]:
    """Duplicate keys silently shadow one another at compile time."""
    data = discover_bib(ctx)
    if not data.files:
        return []
    counts: dict[str, int] = {}
    for entry in data.entries:
        counts[entry.key] = counts.get(entry.key, 0) + 1
    return [
        ctx.finding(
            "bib.duplicate-keys",
            Severity.ERROR,
            f"bib key {key!r} is defined {count} times",
            rule="BIB.duplicate-keys",
            suggestion="keep a single definition per key",
        )
        for key, count in sorted(counts.items())
        if count > 1
    ]


@register("bib.missing-fields", "Entries carry the core fields", "BIB")
def bib_missing_fields(ctx: CheckContext) -> list[Finding]:
    """Missing author/title/year breaks the reference and the reader's trust."""
    data = discover_bib(ctx)
    if not data.files:
        return []
    findings: list[Finding] = []
    for entry in data.entries:
        missing = []
        if "author" not in entry.fields and "editor" not in entry.fields:
            if not (entry.entry_type == "misc"):
                missing.append("author")
        if "title" not in entry.fields:
            missing.append("title")
        if "year" not in entry.fields and "date" not in entry.fields:
            missing.append("year")
        if missing:
            findings.append(
                ctx.finding(
                    "bib.missing-fields",
                    Severity.WARNING,
                    f"entry {entry.key!r} is missing: {', '.join(missing)}",
                    rule="BIB.missing-fields",
                    suggestion="complete the bibliographic record",
                )
            )
    return findings


@register("bib.uncited", "No dead bibliography entries", "BIB")
def bib_uncited(ctx: CheckContext) -> list[Finding]:
    """Uncited entries bloat the .bib and sometimes signal a stale draft."""
    data = discover_bib(ctx)
    if not data.files:
        return []
    cited = {key for key, _ in latex.cite_keys(ctx.doc.text)}
    dead = sorted(entry.key for entry in data.entries if entry.key not in cited)
    if not dead:
        return []
    shown = ", ".join(dead[:10])
    more = f" (+{len(dead) - 10} more)" if len(dead) > 10 else ""
    return [
        ctx.finding(
            "bib.uncited",
            Severity.INFO,
            f"{len(dead)} bib entries are never cited: {shown}{more}",
            rule="BIB.uncited",
            suggestion="prune unused entries or cite them",
        )
    ]


@register("bib.staleness", "Literature engagement is recent", "BIB")
def bib_staleness(ctx: CheckContext) -> list[Finding]:
    """Few recent references reads as disengagement from the current literature."""
    data = discover_bib(ctx)
    if not data.files:
        return []
    years = [entry.year for entry in data.entries if entry.year]
    if not years:
        return []
    newest = max(years)
    window = ctx.config.get("bibliography.recent_window", 5)
    need = ctx.config.get("bibliography.min_recent", 3)
    recent = sum(1 for year in years if year >= newest - window + 1)
    if recent >= need:
        return []
    return [
        ctx.finding(
            "bib.staleness",
            Severity.WARNING,
            f"only {recent} reference(s) within {window} years of {newest}; "
            f"the literature engagement looks stale (want >= {need})",
            rule="BIB.staleness",
            suggestion="cite recent work to show current engagement",
        )
    ]
