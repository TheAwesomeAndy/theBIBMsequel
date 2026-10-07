# SPDX-License-Identifier: Apache-2.0
"""Module M5: blinding, ethics, and data-availability compliance.

The blinding mode (venue.blinding) drives which checks run:
- unset: only an informational nudge to configure it;
- double: scrub author identity, acknowledgments, self-reveals, links,
  and metadata;
- single/none: the author block is required.

Ethics and data-availability statements are checked in every mode.
"""
from __future__ import annotations

import re

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

_AUTHOR_ALLOWED = {
    "anonymous", "author", "authors", "anonymized", "blind",
    "review", "submission", "for", "double",
}
URL_RE = re.compile(r"https?://[^\s}\)]+")
URL_CMD_RE = re.compile(r"\\(?:url|href)\s*\{([^{}]+)\}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _author(text: str) -> tuple[str, int] | None:
    return latex.command_arg(text, "author")


def _human_subjects(ctx: CheckContext, lowered_body: str) -> bool:
    if ctx.config.get("venue.human_subjects"):
        return True
    return any(term in lowered_body for term in vocab.HUMAN_SUBJECT_TERMS)


@register("blind.unset", "Blinding mode is configured", "M5")
def blind_unset(ctx: CheckContext) -> list[Finding]:
    """Without a declared blinding mode the compliance gates cannot run."""
    if ctx.config.get("venue.blinding", "unset") != "unset":
        return []
    return [
        ctx.finding(
            "blind.unset",
            Severity.INFO,
            "venue.blinding is unset; set it to 'double', 'single', or 'none' in harness/papercheck.toml",
            rule="M5.blind-unset",
            suggestion="declare the venue blinding policy to enable compliance checks",
        )
    ]


@register("blind.author-block", "Double-blind: author block is anonymized", "M5")
def blind_author_block(ctx: CheckContext) -> list[Finding]:
    """A real name in the author block breaks double-blind review."""
    if ctx.config.get("venue.blinding", "unset") != "double":
        return []
    author = _author(ctx.doc.text)
    if author is None:
        return []
    content, pos = author
    tokens = [word.lower() for word in latex.words(latex.detex(content))]
    if any(token not in _AUTHOR_ALLOWED for token in tokens):
        return [
            ctx.finding(
                "blind.author-block",
                Severity.ERROR,
                "author block contains a real name under double-blind review",
                pos=pos,
                rule="M5.author-block",
                suggestion="replace with 'Anonymous Authors' for submission",
            )
        ]
    return []


@register("blind.acknowledgments", "Double-blind: no acknowledgments or funding", "M5")
def blind_acknowledgments(ctx: CheckContext) -> list[Finding]:
    """Acknowledgments, thanks, and funding lines de-anonymize authors."""
    if ctx.config.get("venue.blinding", "unset") != "double":
        return []
    text = ctx.doc.text
    findings: list[Finding] = []
    for section in latex.sections(text):
        if section.title_matches("acknowledg", "funding"):
            findings.append(
                ctx.finding(
                    "blind.acknowledgments",
                    Severity.ERROR,
                    f"{section.title!r} section de-anonymizes under double-blind review",
                    pos=section.start,
                    rule="M5.acknowledgments",
                    suggestion="remove until the camera-ready version",
                )
            )
    match = re.search(r"\\thanks\s*\{", text)
    if match:
        findings.append(
            ctx.finding(
                "blind.acknowledgments",
                Severity.ERROR,
                "\\thanks note de-anonymizes under double-blind review",
                pos=match.start(),
                rule="M5.acknowledgments",
                suggestion="remove \\thanks until camera-ready",
            )
        )
    return findings


@register("blind.self-reveal", "Double-blind: no self-revealing references", "M5")
def blind_self_reveal(ctx: CheckContext) -> list[Finding]:
    """'Our previous work' points reviewers straight at the authors."""
    if ctx.config.get("venue.blinding", "unset") != "double":
        return []
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    findings: list[Finding] = []
    for phrase in vocab.SELF_REVEAL:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            findings.append(
                ctx.finding(
                    "blind.self-reveal",
                    Severity.WARNING,
                    f"self-revealing phrase {phrase!r} under double-blind review",
                    pos=offset + index,
                    rule="M5.self-reveal",
                    suggestion="cite in the third person ('prior work [X] showed')",
                )
            )
            start = index + 1
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings


@register("blind.links", "Double-blind: no identifiable repository links", "M5")
def blind_links(ctx: CheckContext) -> list[Finding]:
    """An identifiable repo URL leaks author identity."""
    if ctx.config.get("venue.blinding", "unset") != "double":
        return []
    text = ctx.doc.text
    urls: list[tuple[str, int]] = []
    for match in URL_RE.finditer(text):
        urls.append((match.group(0), match.start()))
    for match in URL_CMD_RE.finditer(text):
        urls.append((match.group(1), match.start()))
    findings: list[Finding] = []
    for url, pos in urls:
        lowered = url.lower()
        if any(host in lowered for host in vocab.ANONYMIZED_HOSTS):
            continue
        if any(host in lowered for host in vocab.IDENTIFIABLE_HOSTS):
            findings.append(
                ctx.finding(
                    "blind.links",
                    Severity.ERROR,
                    f"identifiable repository URL leaks author identity: {url}",
                    pos=pos,
                    rule="M5.links",
                    suggestion="use an anonymized mirror (anonymous.4open.science)",
                )
            )
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings


@register("blind.metadata", "Double-blind: no identifying metadata", "M5")
def blind_metadata(ctx: CheckContext) -> list[Finding]:
    """Emails and ORCIDs identify the authors."""
    if ctx.config.get("venue.blinding", "unset") != "double":
        return []
    text = ctx.doc.text
    findings: list[Finding] = []
    email = EMAIL_RE.search(text)
    if email:
        findings.append(
            ctx.finding(
                "blind.metadata",
                Severity.WARNING,
                "an email address appears in the manuscript under double-blind review",
                pos=email.start(),
                rule="M5.metadata",
                suggestion="remove contact details until camera-ready",
            )
        )
    orcid = re.search(r"orcid", text, re.IGNORECASE)
    if orcid:
        findings.append(
            ctx.finding(
                "blind.metadata",
                Severity.WARNING,
                "an ORCID appears in the manuscript under double-blind review",
                pos=orcid.start(),
                rule="M5.metadata",
                suggestion="remove ORCIDs until camera-ready",
            )
        )
    return findings


@register("single.authors-required", "Single-blind/camera: author block present", "M5")
def single_authors_required(ctx: CheckContext) -> list[Finding]:
    """Single-blind and final versions require named authors."""
    if ctx.config.get("venue.blinding", "unset") not in ("single", "none"):
        return []
    author = _author(ctx.doc.text)
    if author is None or not latex.detex(author[0]).strip():
        return [
            ctx.finding(
                "single.authors-required",
                Severity.ERROR,
                "no author block found; single-blind and final versions require named authors",
                rule="M5.authors-required",
                suggestion="add \\author{...} with names and affiliations",
            )
        ]
    return []


@register("ethics.statement", "Ethics statement for human-subject work", "M5")
def ethics_statement(ctx: CheckContext) -> list[Finding]:
    """Human-subject data requires an ethics/consent statement."""
    body, _ = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.ETHICS_TERMS):
        return []
    if ctx.config.get("venue.human_subjects"):
        return [
            ctx.finding(
                "ethics.statement",
                Severity.ERROR,
                "human-subject study without an ethics/consent statement",
                rule="M5.ethics",
                suggestion="add IRB approval and informed-consent language",
            )
        ]
    if any(term in lowered for term in vocab.HUMAN_SUBJECT_TERMS):
        return [
            ctx.finding(
                "ethics.statement",
                Severity.WARNING,
                "human-subject terms present but no ethics statement found",
                rule="M5.ethics",
                suggestion="add an ethics statement, or set venue.human_subjects to clarify",
            )
        ]
    return []


@register("data.availability", "Data availability statement present", "M5")
def data_availability(ctx: CheckContext) -> list[Finding]:
    """A visible data-availability statement is mandatory for Q1 venues."""
    body, _ = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.DATA_AVAILABILITY_TERMS):
        return []
    return [
        ctx.finding(
            "data.availability",
            Severity.ERROR,
            "no data availability statement found",
            rule="M5.data",
            suggestion="add a Data Availability statement (repository, request policy, or restriction)",
        )
    ]
