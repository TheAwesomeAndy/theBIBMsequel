# SPDX-License-Identifier: Apache-2.0
"""Module M4B: introduction micro-rules.

- An introduction section must exist in a sectioned manuscript.
- The roadmap paragraph ("The rest of this paper is organized as
  follows") is forbidden anywhere in the body.
- The opening must not lead with a cliche ("In recent years...").
- The research gap needs compound framing: at least two gap archetypes.
- A contributions list (itemize/enumerate) must appear in the
  introduction and forward-reference the sections that deliver it.
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

# How many characters of the detexed introduction opening are scanned for
# cliche hooks.
_OPENING_WINDOW = 300

# How many characters before a list environment are scanned for a
# "contributions" lead-in ("Our contributions are:").
_LEAD_IN_WINDOW = 200

# \ref/\autoref/\cref/\Cref with an argument starting "sec".
_SECTION_REF = re.compile(r"\\(?:ref|autoref|[cC]ref)\s*\{\s*[sS]ec")
# The literal word "Section" followed by whitespace/~ and a number or a
# ref-like command.
_SECTION_WORD = re.compile(r"\bSection[\s~]+(?:\d|\\(?:ref|autoref|[cC]ref)\b)")


def _introduction(text: str) -> latex.Section | None:
    return latex.find_section(latex.sections(text), "introduction")


def _contribution_lists(ctx: CheckContext, intro: latex.Section) -> list[latex.Environment]:
    """itemize/enumerate blocks inside the intro that mention contributions.

    A list counts as a contributions list when its body, or the
    ``_LEAD_IN_WINDOW`` characters before it, mention "contribut"
    (contribution/contribute).
    """
    text = ctx.doc.text
    found: list[latex.Environment] = []
    for env in latex.environments(text, "itemize", "enumerate"):
        if not intro.body_start <= env.start < intro.body_end:
            continue
        lead_in = text[max(0, env.start - _LEAD_IN_WINDOW) : env.start]
        if "contribut" in env.body.lower() or "contribut" in lead_in.lower():
            found.append(env)
    return found


@register("intro.present", "An introduction section exists", "M4B")
def intro_present(ctx: CheckContext) -> list[Finding]:
    """A sectioned manuscript needs a section titled like an introduction."""
    all_sections = latex.sections(ctx.doc.text)
    if not all_sections:
        return []
    if latex.find_section(all_sections, "introduction") is None:
        return [
            ctx.finding(
                "intro.present",
                Severity.WARNING,
                "document has sections but none titled like an introduction",
                pos=all_sections[0].start,
                rule="M4B.intro-present",
            )
        ]
    return []


@register("intro.roadmap", "No roadmap paragraph", "M4B")
def intro_roadmap(ctx: CheckContext) -> list[Finding]:
    """The 'rest of this paper is organized as follows' paragraph is forbidden."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    findings: list[Finding] = []
    for phrase in vocab.ROADMAP:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            findings.append(
                ctx.finding(
                    "intro.roadmap",
                    Severity.ERROR,
                    f"forbidden roadmap phrase {phrase!r}",
                    pos=offset + index,
                    rule="M4B.roadmap",
                    suggestion="cut the roadmap paragraph; let section titles carry the structure",
                )
            )
            start = index + 1
    return findings


@register("intro.cliche-opening", "Introduction avoids cliche openings", "M4B")
def intro_cliche_opening(ctx: CheckContext) -> list[Finding]:
    """'In recent years...' openings waste the page-1 hook."""
    intro = _introduction(ctx.doc.text)
    if intro is None:
        return []
    opening = latex.detex(intro.body).lower()[:_OPENING_WINDOW]
    findings: list[Finding] = []
    for phrase in vocab.CLICHE_OPENINGS:
        if phrase in opening:
            findings.append(
                ctx.finding(
                    "intro.cliche-opening",
                    Severity.WARNING,
                    f"introduction opens with cliche {phrase!r}",
                    pos=intro.body_start,
                    rule="M4B.cliche-opening",
                    suggestion="open with the specific problem or a striking fact",
                )
            )
    return findings


@register("intro.gap", "Introduction states a compound research gap", "M4B")
def intro_gap(ctx: CheckContext) -> list[Finding]:
    """A compound research gap combines at least two gap archetypes."""
    intro = _introduction(ctx.doc.text)
    if intro is None:
        return []
    lowered = intro.body.lower()
    matched = [
        name
        for name, phrases in vocab.GAP_ARCHETYPES.items()
        if any(phrase in lowered for phrase in phrases)
    ]
    if len(matched) >= 2:
        return []
    archetypes = ", ".join(vocab.GAP_ARCHETYPES)
    if not matched:
        return [
            ctx.finding(
                "intro.gap",
                Severity.WARNING,
                "no research-gap language found in the introduction",
                pos=intro.body_start,
                rule="M4B.gap",
                suggestion=f"state the gap using at least two archetypes: {archetypes}",
            )
        ]
    return [
        ctx.finding(
            "intro.gap",
            Severity.WARNING,
            f"only one gap archetype ({matched[0]}) found in the introduction; "
            f"a compound gap needs at least two of: {archetypes}",
            pos=intro.body_start,
            rule="M4B.gap",
        )
    ]


@register("intro.contributions", "Introduction has an explicit contributions list", "M4B")
def intro_contributions(ctx: CheckContext) -> list[Finding]:
    """The introduction should enumerate contributions as an itemized list."""
    intro = _introduction(ctx.doc.text)
    if intro is None:
        return []
    if _contribution_lists(ctx, intro):
        return []
    return [
        ctx.finding(
            "intro.contributions",
            Severity.WARNING,
            "no contributions list (itemize/enumerate mentioning 'contribution') "
            "found in the introduction",
            pos=intro.body_start,
            rule="M4B.contributions",
            suggestion="end the introduction with an itemized contributions list",
        )
    ]


@register("intro.forward-refs", "Contributions list forward-references sections", "M4B")
def intro_forward_refs(ctx: CheckContext) -> list[Finding]:
    """Each contribution should point to the section that delivers it."""
    intro = _introduction(ctx.doc.text)
    if intro is None:
        return []
    findings: list[Finding] = []
    for env in _contribution_lists(ctx, intro):
        if _SECTION_REF.search(env.body) or _SECTION_WORD.search(env.body):
            continue
        findings.append(
            ctx.finding(
                "intro.forward-refs",
                Severity.WARNING,
                "contributions list contains no forward reference to a section",
                pos=env.start,
                rule="M4B.forward-refs",
                suggestion="anchor each contribution with \\ref{sec:...} or Section~N",
            )
        )
    return findings
