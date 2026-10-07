# SPDX-License-Identifier: Apache-2.0
"""Core data model: findings, severities, document, and check context."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"error": 2, "warning": 1, "info": 0}[self.value]


@dataclass
class Finding:
    """A single reported issue, tied to a check id and a source location."""

    check_id: str
    severity: Severity
    message: str
    path: str = ""
    line: int = 0
    rule: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "rule": self.rule,
            "suggestion": self.suggestion,
        }

    def format_text(self) -> str:
        location = f"{self.path}:{self.line}: " if self.path else ""
        suffix = f" (try: {self.suggestion})" if self.suggestion else ""
        return f"[{self.severity.value.upper()}] {self.check_id}: {location}{self.message}{suffix}"


@dataclass
class Segment:
    """A contiguous chunk of the expanded document mapping back to one file."""

    out_start: int
    out_end: int
    path: str
    file_text: str
    file_offset: int


class SourceMap:
    """Maps positions in the expanded document back to (path, line)."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self._starts: list[int] = []

    def add(self, segment: Segment) -> None:
        self.segments.append(segment)
        self._starts.append(segment.out_start)

    def resolve(self, pos: int) -> tuple[str, int]:
        if not self.segments:
            return "", 0
        index = bisect.bisect_right(self._starts, pos) - 1
        index = max(0, min(index, len(self.segments) - 1))
        segment = self.segments[index]
        offset = segment.file_offset + max(0, min(pos, segment.out_end) - segment.out_start)
        offset = min(offset, len(segment.file_text))
        line = segment.file_text.count("\n", 0, offset) + 1
        return segment.path, line


@dataclass
class Document:
    """A comment-stripped, input-expanded LaTeX document."""

    main_path: Path
    text: str
    source_map: SourceMap = field(default_factory=SourceMap)

    def location(self, pos: int) -> tuple[str, int]:
        path, line = self.source_map.resolve(pos)
        if not path:
            return str(self.main_path), self.text.count("\n", 0, pos) + 1
        return path, line


@dataclass
class CheckContext:
    """Everything a check needs: the document, config, and repo root."""

    doc: Document
    config: "Config"
    root: Path

    def finding(
        self,
        check_id: str,
        severity: Severity,
        message: str,
        pos: int | None = None,
        rule: str = "",
        suggestion: str = "",
    ) -> Finding:
        path, line = ("", 0)
        if pos is not None:
            path, line = self.doc.location(pos)
        else:
            path = str(self.doc.main_path)
        return Finding(
            check_id=check_id,
            severity=severity,
            message=message,
            path=path,
            line=line,
            rule=rule,
            suggestion=suggestion,
        )


# Imported at the bottom to avoid a circular import at type-checking time.
from papercheck.config import Config  # noqa: E402  pylint: disable=wrong-import-position
