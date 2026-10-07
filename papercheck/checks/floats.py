# SPDX-License-Identifier: Apache-2.0
"""Module M4D-VIS: figure and table discipline.

- Every float carries a caption and a label.
- Every labelled float is referenced in the prose, and introduced in the
  text before it appears.
- Prose never distinguishes graphics by color alone (grayscale printing
  and colorblind readers).
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

CAPTION_RE = re.compile(r"\\caption\b")
LABEL_RE = re.compile(r"\\label\s*\{([^{}]+)\}")
REF_RE = re.compile(r"\\(?:ref|autoref|cref|Cref|pageref|eqref)\s*\{([^{}]+)\}")


def _labels(body: str) -> list[str]:
    return [match.group(1).strip() for match in LABEL_RE.finditer(body)]


def _reference_positions(text: str) -> dict[str, list[int]]:
    """Map every referenced label key to the positions it is cited at."""
    positions: dict[str, list[int]] = {}
    for match in REF_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                positions.setdefault(key, []).append(match.start())
    return positions


@register("floats.caption", "Every float has a caption", "M4D-VIS")
def floats_caption(ctx: CheckContext) -> list[Finding]:
    """A figure or table without a caption is unreadable out of context."""
    findings: list[Finding] = []
    for env in latex.environments(ctx.doc.text, "figure", "table"):
        if not CAPTION_RE.search(env.body):
            findings.append(
                ctx.finding(
                    "floats.caption",
                    Severity.ERROR,
                    f"{env.name} environment has no \\caption",
                    pos=env.start,
                    rule="M4D-VIS.caption",
                    suggestion="add a self-contained \\caption",
                )
            )
    return findings


@register("floats.label", "Every float has a label", "M4D-VIS")
def floats_label(ctx: CheckContext) -> list[Finding]:
    """Without a label a float cannot be cross-referenced."""
    findings: list[Finding] = []
    for env in latex.environments(ctx.doc.text, "figure", "table"):
        if not LABEL_RE.search(env.body):
            findings.append(
                ctx.finding(
                    "floats.label",
                    Severity.WARNING,
                    f"{env.name} environment has no \\label",
                    pos=env.start,
                    rule="M4D-VIS.label",
                    suggestion="add a \\label so the float can be referenced",
                )
            )
    return findings


@register("floats.referenced", "Every labelled float is referenced", "M4D-VIS")
def floats_referenced(ctx: CheckContext) -> list[Finding]:
    """A float nobody refers to is dead weight; the reader never sees it in flow."""
    refs = _reference_positions(ctx.doc.text)
    findings: list[Finding] = []
    for env in latex.environments(ctx.doc.text, "figure", "table"):
        for key in _labels(env.body):
            if key not in refs:
                findings.append(
                    ctx.finding(
                        "floats.referenced",
                        Severity.ERROR,
                        f"float label {key!r} is never referenced in the text",
                        pos=env.start,
                        rule="M4D-VIS.referenced",
                        suggestion="introduce the float in the prose with a \\ref",
                    )
                )
    return findings


@register("floats.ref-before", "Floats are introduced before they appear", "M4D-VIS")
def floats_ref_before(ctx: CheckContext) -> list[Finding]:
    """The first reference to a float should precede the float itself."""
    refs = _reference_positions(ctx.doc.text)
    findings: list[Finding] = []
    for env in latex.environments(ctx.doc.text, "figure", "table"):
        for key in _labels(env.body):
            positions = refs.get(key)
            if not positions:
                continue
            if min(positions) > env.end:
                findings.append(
                    ctx.finding(
                        "floats.ref-before",
                        Severity.WARNING,
                        f"float {key!r} appears before it is first referenced in the text",
                        pos=env.start,
                        rule="M4D-VIS.ref-before",
                        suggestion="mention the float in the prose before it appears",
                    )
                )
    return findings


# (a) "red line", (b) "shown in red".
_COLOR_ADJACENT = re.compile(
    r"\b(" + "|".join(vocab.COLOR_WORDS) + r")\s+(" + "|".join(vocab.COLOR_TARGETS) + r")\b"
)
_COLOR_SHOWN_IN = re.compile(
    r"\b(?:shown|depicted|plotted|marked|highlighted|indicated)\s+in\s+(" + "|".join(vocab.COLOR_WORDS) + r")\b"
)


@register("floats.color-only", "Graphics are not distinguished by color alone", "M4D-VIS")
def floats_color_only(ctx: CheckContext) -> list[Finding]:
    """Color-only references fail on grayscale printouts and for colorblind readers."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    findings: list[Finding] = []
    seen: set[int] = set()
    for pattern in (_COLOR_ADJACENT, _COLOR_SHOWN_IN):
        for match in pattern.finditer(lowered):
            pos = offset + match.start()
            if pos in seen:
                continue
            seen.add(pos)
            findings.append(
                ctx.finding(
                    "floats.color-only",
                    Severity.WARNING,
                    f"color-only reference {match.group(0)!r}; illegible on grayscale printouts "
                    "and for colorblind readers",
                    pos=pos,
                    rule="M4D-VIS.color-only",
                    suggestion="also distinguish by stroke pattern, marker shape, or a direct label",
                )
            )
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings
