# SPDX-License-Identifier: Apache-2.0
"""Configuration: built-in defaults overlaid with a TOML file and CLI options.

The config file is TOML (stdlib ``tomllib``; no third-party dependencies).
Default location: ``harness/papercheck.toml`` relative to the repo root.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

PROFILES = ("draft", "submission", "camera")

DEFAULTS: dict[str, Any] = {
    "paper": {
        "main_tex": "manuscript/main.tex",
        # Extra .bib files to scan. Files referenced by \bibliography{...}
        # and .bib files next to the main tex file are found automatically.
        "bib": [],
    },
    "venue": {
        "name": "",
        # "double" | "single" | "none" | "unset"
        "blinding": "unset",
        # True when the study involves human participants (enables the
        # leakage and ethics checks at full strength).
        "human_subjects": False,
    },
    "title": {"min_words": 8, "max_words": 13},
    "abstract": {"max_words": 200, "min_sentences": 4, "max_sentences": 4},
    "keywords": {"min": 4, "max": 6, "required": False},
    "language": {
        # Em dashes are a common machine-writing tell; allow opting out.
        "allow_em_dash": False,
    },
    "bibliography": {
        # Warn when fewer than this many references were published within
        # recent_window years of the newest reference in the .bib file.
        "min_recent": 3,
        "recent_window": 5,
    },
    "related_work": {
        # When true, a Related Work section placed before Methods is a
        # warning (the playbook prefers postponed synthesis).
        "prefer_postponed": True,
    },
    "cover_letter": {"max_words": 400, "paragraphs": 3},
    "checks": {
        # Check ids (or dotted prefixes) to disable entirely.
        "disable": [],
        # Check ids (or dotted prefixes) downgraded to warnings.
        "warn_only": [],
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Nested config with dotted-path access: ``config.get("title.max_words")``."""

    def __init__(self, data: dict[str, Any] | None = None, profile: str = "draft") -> None:
        self.data = _deep_merge(DEFAULTS, data or {})
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile!r} (expected one of {PROFILES})")
        self.profile = profile

    @classmethod
    def load(cls, path: Path | None, profile: str = "draft") -> "Config":
        data: dict[str, Any] = {}
        if path is not None and path.exists():
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        return cls(data, profile=profile)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def strict_warnings(self) -> bool:
        """Submission and camera profiles fail on warnings, not just errors."""
        return self.profile in ("submission", "camera")

    def is_disabled(self, check_id: str) -> bool:
        return _matches_any(check_id, self.get("checks.disable", []))

    def is_warn_only(self, check_id: str) -> bool:
        return _matches_any(check_id, self.get("checks.warn_only", []))


def _matches_any(check_id: str, patterns: list[str]) -> bool:
    for pattern in patterns or []:
        if check_id == pattern or check_id.startswith(pattern.rstrip(".") + "."):
            return True
    return False
