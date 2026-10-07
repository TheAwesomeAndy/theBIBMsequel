# SPDX-License-Identifier: Apache-2.0
"""Modules M3/M4E/M4F: global structure, limitations, related-work tone.

- The paper carries the expected structural skeleton.
- Limitations are stated (owned, not hidden) and each is followed by a
  defense or a future-work pivot.
- The conclusion is not a copy of the abstract.
- Related work is postponed, generous, and synthesized rather than
  enumerated.
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

METHODS_KEYWORDS = ("method", "methodology", "materials", "experimental setup", "approach")
RESULTS_KEYWORDS = ("result", "findings", "experiments")
DISCUSSION_KEYWORDS = ("discussion",)
CONCLUSION_KEYWORDS = ("conclusion", "concluding")
RELATED_KEYWORDS = ("related work", "prior work", "literature review", "background")


def _first(all_sections: list[latex.Section], *keywords: str) -> latex.Section | None:
    for section in all_sections:
        if section.title_matches(*keywords):
            return section
    return None


def _find_all(text: str, phrase: str) -> list[int]:
    positions = []
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            break
        positions.append(index)
        start = index + 1
    return positions


def _normalized_sentences(text: str) -> list[str]:
    plain = latex.detex(text).lower()
    result = []
    for sentence in latex.sentences(plain):
        words = latex.words(sentence)
        if words:
            result.append(" ".join(words))
    return result


@register("structure.sections", "Paper has the expected structural skeleton", "M3")
def structure_sections(ctx: CheckContext) -> list[Finding]:
    """A sectioned paper should have methods, results, discussion, conclusion."""
    all_sections = latex.sections(ctx.doc.text)
    if len(all_sections) < 2:
        return []
    missing = []
    if not _first(all_sections, *METHODS_KEYWORDS):
        missing.append("methods")
    if not _first(all_sections, *RESULTS_KEYWORDS):
        missing.append("results")
    if not _first(all_sections, *DISCUSSION_KEYWORDS):
        missing.append("discussion")
    if not _first(all_sections, *CONCLUSION_KEYWORDS):
        missing.append("conclusion")
    if not missing:
        return []
    return [
        ctx.finding(
            "structure.sections",
            Severity.WARNING,
            f"missing expected section(s): {', '.join(missing)}",
            pos=all_sections[0].start,
            rule="M3.sections",
            suggestion="a combined 'Results and Discussion' section satisfies both slots",
        )
    ]


@register("limits.present", "Limitations are stated explicitly", "M4F")
def limits_present(ctx: CheckContext) -> list[Finding]:
    """Owning the weaknesses is a Q1 requirement; hidden limitations get caught."""
    body, _ = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.LIMITATION_TERMS):
        return []
    return [
        ctx.finding(
            "limits.present",
            Severity.ERROR,
            "no limitation language found; own the weaknesses and state 1-3 limitations",
            rule="M4F.limits",
            suggestion="add a Limitations paragraph stating boundary conditions",
        )
    ]


@register("limits.pivot", "Each limitation has a defense or pivot", "M4F")
def limits_pivot(ctx: CheckContext) -> list[Finding]:
    """A limitation without a follow-up reads as a concession, not a boundary."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    hits = [lowered.find(term) for term in vocab.LIMITATION_TERMS if lowered.find(term) >= 0]
    if not hits:
        return []
    first = min(hits)
    window = lowered[first : first + 600]
    if any(term in window for term in vocab.PIVOT_TERMS):
        return []
    return [
        ctx.finding(
            "limits.pivot",
            Severity.WARNING,
            "limitation stated without a nearby defense or future-work pivot",
            pos=offset + first,
            rule="M4F.pivot",
            suggestion="follow the limitation with a defense, a trade-off, or a future-work pointer",
        )
    ]


@register("conclusion.abstract-copy", "Conclusion is not a copy of the abstract", "M4F")
def conclusion_abstract_copy(ctx: CheckContext) -> list[Finding]:
    """A recycled abstract wastes the conclusion's chance to land the message."""
    abstract = latex.get_abstract(ctx.doc.text)
    conclusion = _first(latex.sections(ctx.doc.text), *CONCLUSION_KEYWORDS)
    if abstract is None or conclusion is None:
        return []
    abstract_sentences = set(_normalized_sentences(abstract[0]))
    conclusion_sentences = _normalized_sentences(conclusion.body)
    if len(conclusion_sentences) < 2 or not abstract_sentences:
        return []
    overlap = sum(1 for sentence in conclusion_sentences if sentence in abstract_sentences)
    if overlap / len(conclusion_sentences) >= 0.5:
        return [
            ctx.finding(
                "conclusion.abstract-copy",
                Severity.ERROR,
                f"{overlap}/{len(conclusion_sentences)} conclusion sentences are copied verbatim "
                "from the abstract",
                pos=conclusion.body_start,
                rule="M4F.abstract-copy",
                suggestion="rewrite the conclusion around takeaways and next steps",
            )
        ]
    return []


@register("related.position", "Related work is postponed", "M4E")
def related_position(ctx: CheckContext) -> list[Finding]:
    """A dense literature sandbar on page 1 fatigues the reader early."""
    if not ctx.config.get("related_work.prefer_postponed"):
        return []
    all_sections = latex.sections(ctx.doc.text)
    related = _first(all_sections, *RELATED_KEYWORDS)
    methods = _first(all_sections, *METHODS_KEYWORDS)
    if related is None or methods is None:
        return []
    if related.start < methods.start:
        return [
            ctx.finding(
                "related.position",
                Severity.WARNING,
                "related work appears before the methods; postpone exhaustive synthesis",
                pos=related.start,
                rule="M4E.position",
                suggestion="move the literature synthesis to late in the paper (the sandbar rule)",
            )
        ]
    return []


@register("related.hostile", "Related work credits prior work generously", "M4E")
def related_hostile(ctx: CheckContext) -> list[Finding]:
    """Credit is like love: hostile take-downs cost goodwill, not gain it."""
    related = _first(latex.sections(ctx.doc.text), *RELATED_KEYWORDS)
    if related is None:
        return []
    lowered = related.body.lower()
    suggestion = "acknowledge generously (" + ", ".join(vocab.GENEROUS[:3]) + ")"
    findings: list[Finding] = []
    for phrase in vocab.HOSTILE:
        if " " in phrase:
            positions = _find_all(lowered, phrase)
        else:
            positions = [match.start() for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lowered)]
        for index in positions:
            findings.append(
                ctx.finding(
                    "related.hostile",
                    Severity.WARNING,
                    f"hostile phrasing {phrase!r} toward prior work",
                    pos=related.body_start + index,
                    rule="M4E.hostile",
                    suggestion=suggestion,
                )
            )
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings


@register("related.laundry-list", "Related work synthesizes rather than enumerates", "M4E")
def related_laundry_list(ctx: CheckContext) -> list[Finding]:
    """Three-plus consecutive citation-led sentences is a laundry list, not synthesis."""
    related = _first(latex.sections(ctx.doc.text), *RELATED_KEYWORDS)
    if related is None:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", related.body)
    run = 0
    best = 0
    cite_lead = re.compile(r"\\[cC]ite")
    author_lead = re.compile(r"^[A-Z][A-Za-z]+\s+et\s+al")
    for chunk in chunks:
        head = chunk.strip()[:60]
        if cite_lead.search(head) or author_lead.search(head):
            run += 1
            best = max(best, run)
        else:
            run = 0
    if best >= 3:
        return [
            ctx.finding(
                "related.laundry-list",
                Severity.WARNING,
                f"{best} consecutive citation-led sentences read as a laundry list",
                pos=related.body_start,
                rule="M4E.laundry-list",
                suggestion="group references to support an argument, not 'A did X. B did Y.'",
            )
        ]
    return []
