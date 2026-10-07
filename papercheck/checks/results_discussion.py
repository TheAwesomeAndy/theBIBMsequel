# SPDX-License-Identifier: Apache-2.0
"""Module M4D: results/discussion discipline.

- Results state observed facts; speculative language belongs in the
  Discussion.
- A discussion-like section must exist (a combined "Results and
  Discussion" section counts).
- The Discussion must compare with the literature (RCI matrix) via
  explicit citations.
- Interpretation needs cautious semantics (suggests/indicates/may).
- Absolute claims ("proves that", "undoubtedly") are forbidden anywhere;
  known overclaim phrases get a concrete cautious replacement.
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

# Section titles that identify a results-like section.
RESULTS_SECTION_KEYWORDS = ("result", "findings", "experiments")

# Section titles that identify a discussion-like section.
DISCUSSION_SECTION_KEYWORDS = ("discussion",)


def _results_sections(all_sections: list[latex.Section]) -> list[latex.Section]:
    """Pure results-like sections.

    A combined section ("Results and Discussion") is treated as a
    discussion for speculation purposes and is therefore excluded here.
    """
    return [
        section
        for section in all_sections
        if section.title_matches(*RESULTS_SECTION_KEYWORDS)
        and not section.title_matches(*DISCUSSION_SECTION_KEYWORDS)
    ]


def _discussion_sections(all_sections: list[latex.Section]) -> list[latex.Section]:
    """Discussion-like sections, including combined Results-and-Discussion."""
    return [section for section in all_sections if section.title_matches(*DISCUSSION_SECTION_KEYWORDS)]


def _occurrences(lowered: str, phrase: str) -> list[int]:
    """All positions of a lowercase phrase in already-lowercased text.

    Single-word entries ("obviously", "undoubtedly") use word-boundary
    matching so they never fire inside longer words; multi-word phrases
    use plain substring search.
    """
    if " " not in phrase:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")
        return [match.start() for match in pattern.finditer(lowered)]
    positions: list[int] = []
    start = 0
    while True:
        index = lowered.find(phrase, start)
        if index < 0:
            break
        positions.append(index)
        start = index + 1
    return positions


@register("results.speculation", "Results state facts, not speculation", "M4D")
def results_speculation(ctx: CheckContext) -> list[Finding]:
    """Speculative language inside a pure Results section is misplaced."""
    all_sections = latex.sections(ctx.doc.text)
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for section in _results_sections(all_sections):
        lowered = section.body.lower()
        for phrase in vocab.SPECULATION:
            for index in _occurrences(lowered, phrase):
                pos = section.body_start + index
                if (phrase, pos) in seen:
                    continue
                seen.add((phrase, pos))
                findings.append(
                    ctx.finding(
                        "results.speculation",
                        Severity.WARNING,
                        f"speculative phrase {phrase!r} in the results section; "
                        "speculation belongs in the Discussion; Results state observed facts",
                        pos=pos,
                        rule="M4D.speculation",
                        suggestion="move the interpretation to the Discussion",
                    )
                )
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings


@register("discussion.present", "A discussion-like section exists", "M4D")
def discussion_present(ctx: CheckContext) -> list[Finding]:
    """A sectioned manuscript needs a Discussion (or Results and Discussion)."""
    all_sections = latex.sections(ctx.doc.text)
    if not all_sections:
        return []
    if _discussion_sections(all_sections):
        return []
    return [
        ctx.finding(
            "discussion.present",
            Severity.WARNING,
            "document has sections but none titled like a discussion",
            pos=all_sections[0].start,
            rule="M4D.present",
            suggestion="add a Discussion (or Results and Discussion) section interpreting the findings",
        )
    ]


@register("discussion.compare", "Discussion compares with the literature", "M4D")
def discussion_compare(ctx: CheckContext) -> list[Finding]:
    """Findings must be related to prior work with explicit citations."""
    discussions = _discussion_sections(latex.sections(ctx.doc.text))
    if not discussions:
        return []
    if any(latex.cite_keys(section.body) for section in discussions):
        return []
    return [
        ctx.finding(
            "discussion.compare",
            Severity.WARNING,
            "discussion contains no \\cite command; "
            "the RCI matrix requires comparison with the literature",
            pos=discussions[0].body_start,
            rule="M4D.compare",
            suggestion="relate each finding to prior work with explicit citations",
        )
    ]


@register("discussion.hedging", "Discussion interprets with cautious verbs", "M4D")
def discussion_hedging(ctx: CheckContext) -> list[Finding]:
    """A Discussion without hedging language reads as overconfident."""
    discussions = _discussion_sections(latex.sections(ctx.doc.text))
    if not discussions:
        return []
    for section in discussions:
        lowered = section.body.lower()
        if any(verb in lowered for verb in vocab.CAUTIOUS_VERBS):
            return []
    return [
        ctx.finding(
            "discussion.hedging",
            Severity.WARNING,
            "no cautious verbs found in the discussion; "
            "interpretation needs cautious semantics (suggests/indicates/may)",
            pos=discussions[0].body_start,
            rule="M4D.hedging",
            suggestion="hedge interpretations with suggests/indicates/may/appears to",
        )
    ]


@register("claims.overclaim", "No absolute claims anywhere in the body", "M4D")
def claims_overclaim(ctx: CheckContext) -> list[Finding]:
    """Science reports evidence, not certainty; absolute claims are errors."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    findings: list[Finding] = []
    for phrase in vocab.OVERCLAIM:
        for index in _occurrences(lowered, phrase):
            findings.append(
                ctx.finding(
                    "claims.overclaim",
                    Severity.ERROR,
                    f"absolute claim {phrase!r}; science reports evidence, not certainty",
                    pos=offset + index,
                    rule="M4D.overclaim",
                    suggestion="soften to evidence language (suggests/indicates/supports)",
                )
            )
    for phrase, replacement in vocab.CAUTIOUS_REPLACEMENTS.items():
        for index in _occurrences(lowered, phrase):
            findings.append(
                ctx.finding(
                    "claims.overclaim",
                    Severity.ERROR,
                    f"overclaim {phrase!r} needs a cautious rewrite",
                    pos=offset + index,
                    rule="M4D.overclaim",
                    suggestion=replacement,
                )
            )
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings
