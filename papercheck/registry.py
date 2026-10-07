# SPDX-License-Identifier: Apache-2.0
"""Check registry: every check registers itself with a stable dotted id."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from papercheck.model import CheckContext, Finding

CheckFunc = Callable[[CheckContext], list[Finding]]


@dataclass(frozen=True)
class CheckMeta:
    check_id: str
    title: str
    module: str
    description: str
    func: CheckFunc


_REGISTRY: dict[str, CheckMeta] = {}


def register(check_id: str, title: str, module: str, description: str = "") -> Callable[[CheckFunc], CheckFunc]:
    """Decorator registering a check function under a stable id.

    ``module`` names the harness module the check enforces (e.g. "M4A" for
    the title/abstract micro-rules) so reports and docs can group findings.
    """

    def wrap(func: CheckFunc) -> CheckFunc:
        if check_id in _REGISTRY:
            raise ValueError(f"duplicate check id: {check_id}")
        _REGISTRY[check_id] = CheckMeta(
            check_id=check_id,
            title=title,
            module=module,
            description=description or (func.__doc__ or "").strip(),
            func=func,
        )
        return func

    return wrap


def all_checks() -> dict[str, CheckMeta]:
    # Importing the checks package populates the registry.
    import papercheck.checks  # noqa: F401  pylint: disable=unused-import

    return dict(_REGISTRY)


def select_checks(select: list[str] | None, skip: list[str] | None) -> list[CheckMeta]:
    """Resolve --select/--skip patterns (exact ids or dotted prefixes)."""

    def matches(check_id: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if check_id == pattern or check_id.startswith(pattern.rstrip(".") + "."):
                return True
        return False

    metas = sorted(all_checks().values(), key=lambda meta: meta.check_id)
    if select:
        metas = [meta for meta in metas if matches(meta.check_id, select)]
    if skip:
        metas = [meta for meta in metas if not matches(meta.check_id, skip)]
    return metas
