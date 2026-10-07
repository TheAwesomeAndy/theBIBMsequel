# SPDX-License-Identifier: Apache-2.0
"""Modules L1/M4D: hype, overclaim wording, and machine-writing tells.

All scans run over the document body, lowercased, positions mapped back
to the expanded document. Findings are capped per check so a single
systemic problem does not flood the report.
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

CAP = 20

# Markers that make the word "significant" a defensible statistical claim.
STAT_MARKERS = (
    "p <", "p<", "p =", "p=", "p-value", "confidence interval", "t-test",
    "anova", "wilcoxon", "mann-whitney", "permutation test", "chi-square",
    "effect size", "statistically",
)


def _phrase_positions(lowered: str, phrase: str) -> list[int]:
    if " " not in phrase and phrase.isalpha():
        return [match.start() for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lowered)]
    # A left word boundary stops "very " matching inside "every ". The
    # phrase supplies its own right boundary (trailing space or non-word).
    left = r"(?<![A-Za-z])" if phrase[:1].isalpha() else ""
    return [match.start() for match in re.finditer(left + re.escape(phrase), lowered)]


def _scan(
    ctx: CheckContext,
    check_id: str,
    severity: Severity,
    phrases,
    message: str,
    rule: str,
    suggestions: dict | None = None,
) -> list[Finding]:
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    hits: list[tuple[int, str]] = []
    for phrase in phrases:
        for index in _phrase_positions(lowered, phrase):
            hits.append((index, phrase))
    hits.sort()
    findings: list[Finding] = []
    for index, phrase in hits[:CAP]:
        findings.append(
            ctx.finding(
                check_id,
                severity,
                f"{message}: {phrase!r}",
                pos=offset + index,
                rule=rule,
                suggestion=(suggestions or {}).get(phrase, ""),
            )
        )
    if len(hits) > CAP and findings:
        findings[-1].message += f" (+{len(hits) - CAP} more occurrences suppressed)"
    return findings


@register("lang.hype", "No marketing hype", "L1")
def lang_hype(ctx: CheckContext) -> list[Finding]:
    """Marketing superlatives read as a red flag to top-tier reviewers."""
    return _scan(
        ctx, "lang.hype", Severity.ERROR, vocab.HYPE_ERROR,
        "marketing hype triggers desk rejection", "L1.hype",
    )


@register("lang.hype-claims", "Strong claims carry evidence", "L1")
def lang_hype_claims(ctx: CheckContext) -> list[Finding]:
    """Softer hype that needs nearby evidence or removal."""
    return _scan(
        ctx, "lang.hype-claims", Severity.WARNING, vocab.HYPE_WARN,
        "claim needs explicit evidence nearby or removal", "L1.hype-claims",
    )


@register("lang.cautious", "Overclaims are softened to evidence language", "L1")
def lang_cautious(ctx: CheckContext) -> list[Finding]:
    """Known overclaim phrasings get a concrete cautious rewrite."""
    return _scan(
        ctx, "lang.cautious", Severity.ERROR, list(vocab.CAUTIOUS_REPLACEMENTS.keys()),
        "overclaim needs a cautious rewrite", "L1.cautious",
        suggestions=vocab.CAUTIOUS_REPLACEMENTS,
    )


@register("lang.ai-tells", "No machine-writing tells", "L1")
def lang_ai_tells(ctx: CheckContext) -> list[Finding]:
    """Filler phrases that mark generated prose; rewrite in the paper's own voice."""
    return _scan(
        ctx, "lang.ai-tells", Severity.WARNING, vocab.AI_TELLS,
        "machine-writing tell; rewrite in the paper's own voice", "L1.ai-tells",
    )


@register("lang.em-dash", "No em dashes", "L1")
def lang_em_dash(ctx: CheckContext) -> list[Finding]:
    """Em dashes are a common machine-writing tell; opt out via config if intended."""
    if ctx.config.get("language.allow_em_dash"):
        return []
    body, offset = latex.document_body(ctx.doc.text)
    findings: list[Finding] = []
    for match in list(re.finditer("—", body))[:CAP]:
        findings.append(
            ctx.finding(
                "lang.em-dash",
                Severity.WARNING,
                "em dash (set language.allow_em_dash=true to permit)",
                pos=offset + match.start(),
                rule="L1.em-dash",
                suggestion="use a comma, colon, or parentheses",
            )
        )
    return findings


@register("lang.contractions", "No contractions in formal register", "L1")
def lang_contractions(ctx: CheckContext) -> list[Finding]:
    """Formal academic register expands contractions."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    hits: list[int] = []
    for word in vocab.CONTRACTIONS:
        pattern = re.compile(r"(?<![A-Za-z'])" + re.escape(word) + r"(?![A-Za-z'])")
        hits.extend(match.start() for match in pattern.finditer(lowered))
    findings: list[Finding] = []
    for index in sorted(hits)[:CAP]:
        findings.append(
            ctx.finding(
                "lang.contractions",
                Severity.WARNING,
                "contraction in formal prose",
                pos=offset + index,
                rule="L1.contractions",
                suggestion="expand the contraction",
            )
        )
    return findings


@register("lang.weasel", "Quantify instead of intensifying", "L1")
def lang_weasel(ctx: CheckContext) -> list[Finding]:
    """Intensifiers ('very', 'extremely') add emphasis, not information."""
    return _scan(
        ctx, "lang.weasel", Severity.INFO, vocab.WEASEL,
        "intensifier adds emphasis, not information; quantify instead", "L1.weasel",
    )


@register("lang.significant", "'significant' implies a reported test", "M4D")
def lang_significant(ctx: CheckContext) -> list[Finding]:
    """'Significant' without a test reads as a statistical claim you did not make."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(marker in lowered for marker in STAT_MARKERS):
        return []
    findings: list[Finding] = []
    for match in list(re.finditer(r"\bsignificant(?:ly)?\b", lowered))[:CAP]:
        findings.append(
            ctx.finding(
                "lang.significant",
                Severity.WARNING,
                "'significant' implies a statistical test; report the test or say 'substantial'",
                pos=offset + match.start(),
                rule="M4D.significant",
                suggestion="report the test (p-value, effect size) or use 'substantial'",
            )
        )
    return findings
