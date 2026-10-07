# SPDX-License-Identifier: Apache-2.0
"""Module M4A: title, abstract, and keyword micro-rules.

- Title: 8-13 active words, no questions, no filler openers. A colon
  subtitle is allowed (and correlates with citations).
- Abstract: Kent Beck's 4-sentence structure, under 200 words, no
  citations.
- Keywords: 4-6 discoverable terms.
"""
from __future__ import annotations

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register


@register("title.present", "Title exists", "M4A")
def title_present(ctx: CheckContext) -> list[Finding]:
    """The manuscript must declare a title."""
    if latex.get_title(ctx.doc.text) is None:
        return [ctx.finding("title.present", Severity.ERROR, "no \\title{...} found", rule="M4A.title")]
    return []


@register("title.length", "Title is 8-13 words", "M4A")
def title_length(ctx: CheckContext) -> list[Finding]:
    """Force the title into the high-citation 8-13 active-word band."""
    result = latex.get_title(ctx.doc.text)
    if result is None:
        return []
    title, pos = result
    count = len(latex.words(title))
    low = ctx.config.get("title.min_words", 8)
    high = ctx.config.get("title.max_words", 13)
    if count < low or count > high:
        return [
            ctx.finding(
                "title.length",
                Severity.ERROR,
                f"title has {count} words; target {low}-{high}: {title!r}",
                pos=pos,
                rule="M4A.title-length",
            )
        ]
    return []


@register("title.question", "Title is not a question", "M4A")
def title_question(ctx: CheckContext) -> list[Finding]:
    """Question titles underperform; state the finding instead."""
    result = latex.get_title(ctx.doc.text)
    if result is None:
        return []
    title, pos = result
    if "?" in title:
        return [
            ctx.finding(
                "title.question",
                Severity.ERROR,
                f"title is phrased as a question: {title!r}",
                pos=pos,
                rule="M4A.title-question",
                suggestion="state the finding as a declarative claim",
            )
        ]
    return []


@register("title.filler", "Title avoids generic filler openers", "M4A")
def title_filler(ctx: CheckContext) -> list[Finding]:
    """Reject 'A study of...', 'An investigation into...' style openers."""
    result = latex.get_title(ctx.doc.text)
    if result is None:
        return []
    title, pos = result
    lowered = title.lower()
    findings = []
    for prefix in vocab.TITLE_FILLER_PREFIXES:
        if lowered.startswith(prefix):
            findings.append(
                ctx.finding(
                    "title.filler",
                    Severity.WARNING,
                    f"title opens with generic filler {prefix!r}",
                    pos=pos,
                    rule="M4A.title-filler",
                    suggestion="lead with the specific method or finding",
                )
            )
    return findings


@register("abstract.present", "Abstract exists", "M4A")
def abstract_present(ctx: CheckContext) -> list[Finding]:
    """The manuscript must contain an abstract."""
    if latex.get_abstract(ctx.doc.text) is None:
        return [ctx.finding("abstract.present", Severity.ERROR, "no abstract found", rule="M4A.abstract")]
    return []


@register("abstract.length", "Abstract stays under the word limit", "M4A")
def abstract_length(ctx: CheckContext) -> list[Finding]:
    """Abstracts over the limit get skimmed, truncated, or desk-rejected."""
    result = latex.get_abstract(ctx.doc.text)
    if result is None:
        return []
    body, pos = result
    count = len(latex.words(latex.detex(body)))
    limit = ctx.config.get("abstract.max_words", 200)
    if count > limit:
        return [
            ctx.finding(
                "abstract.length",
                Severity.ERROR,
                f"abstract has {count} words; limit is {limit}",
                pos=pos,
                rule="M4A.abstract-length",
            )
        ]
    return []


@register("abstract.structure", "Abstract follows the 4-sentence structure", "M4A")
def abstract_structure(ctx: CheckContext) -> list[Finding]:
    """Kent Beck structure: problem, why it matters, what we achieve, implications."""
    result = latex.get_abstract(ctx.doc.text)
    if result is None:
        return []
    body, pos = result
    count = len(latex.sentences(latex.detex(body)))
    low = ctx.config.get("abstract.min_sentences", 4)
    high = ctx.config.get("abstract.max_sentences", 4)
    if count < low or count > high:
        expected = str(low) if low == high else f"{low}-{high}"
        return [
            ctx.finding(
                "abstract.structure",
                Severity.WARNING,
                f"abstract has {count} sentences; the playbook expects {expected} "
                "(problem, motivation, achievement, implications)",
                pos=pos,
                rule="M4A.abstract-structure",
            )
        ]
    return []


@register("abstract.citations", "Abstract avoids citations", "M4A")
def abstract_citations(ctx: CheckContext) -> list[Finding]:
    """Abstracts circulate detached from the bibliography; keep them self-contained."""
    result = latex.get_abstract(ctx.doc.text)
    if result is None:
        return []
    body, pos = result
    if latex.cite_keys(body):
        return [
            ctx.finding(
                "abstract.citations",
                Severity.WARNING,
                "abstract contains \\cite commands; abstracts should be self-contained",
                pos=pos,
                rule="M4A.abstract-citations",
            )
        ]
    return []


@register("keywords.count", "Keyword count is in the 4-6 discoverability band", "M4A")
def keywords_count(ctx: CheckContext) -> list[Finding]:
    """4-6 keywords balance discoverability against dilution."""
    result = latex.get_keywords(ctx.doc.text)
    if result is None:
        if ctx.config.get("keywords.required", False):
            return [
                ctx.finding(
                    "keywords.count",
                    Severity.WARNING,
                    "no keywords found; add 4-6 search-optimized keywords",
                    rule="M4A.keywords",
                )
            ]
        return []
    keywords, pos = result
    low = ctx.config.get("keywords.min", 4)
    high = ctx.config.get("keywords.max", 6)
    if len(keywords) < low or len(keywords) > high:
        return [
            ctx.finding(
                "keywords.count",
                Severity.WARNING,
                f"found {len(keywords)} keywords; target {low}-{high}: {keywords}",
                pos=pos,
                rule="M4A.keywords-count",
            )
        ]
    return []
