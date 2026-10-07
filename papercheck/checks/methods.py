# SPDX-License-Identifier: Apache-2.0
"""Module M4C: methodology rigor.

- A methods-like section (Methods, Materials, Experimental Setup, ...)
  must exist.
- Evaluation must compare against explicit baselines.
- Human-subject studies must not use pooled cross-validation: identity
  and feature leakage across folds inflates performance.
- Reproducibility details (seeds, versions, hyperparameters, code
  availability) and robustness analyses (ablations, perturbations) must
  appear somewhere in the body.
- Completed work is reported in past tense, not future tense.
"""
from __future__ import annotations

from papercheck import latex, vocab
from papercheck.model import CheckContext, Finding, Severity
from papercheck.registry import register

# Section titles that identify a methods-like section.
METHODS_SECTION_KEYWORDS = (
    "method",
    "methodology",
    "materials",
    "experimental setup",
    "approach",
    "procedure",
)


def _methods_section(text: str) -> latex.Section | None:
    return latex.find_section(latex.sections(text), *METHODS_SECTION_KEYWORDS)


def _first_phrase(lowered: str, phrases: tuple[str, ...]) -> tuple[int, str] | None:
    """Earliest occurrence of any phrase in already-lowercased text."""
    best: tuple[int, str] | None = None
    for phrase in phrases:
        pos = lowered.find(phrase)
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, phrase)
    return best


@register("methods.present", "A methods-like section exists", "M4C")
def methods_present(ctx: CheckContext) -> list[Finding]:
    """A sectioned manuscript must include a methods-like section."""
    all_sections = latex.sections(ctx.doc.text)
    if not all_sections:
        return []
    if latex.find_section(all_sections, *METHODS_SECTION_KEYWORDS) is None:
        return [
            ctx.finding(
                "methods.present",
                Severity.WARNING,
                "no methods-like section found (Methods, Materials, Experimental Setup, Approach, Procedure)",
                rule="M4C.present",
                suggestion="add a Methods section describing the experimental procedure",
            )
        ]
    return []


@register("methods.baseline", "Evaluation compares against explicit baselines", "M4C")
def methods_baseline(ctx: CheckContext) -> list[Finding]:
    """Results without a baseline comparison are uninterpretable."""
    body, _offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.BASELINE_TERMS):
        return []
    return [
        ctx.finding(
            "methods.baseline",
            Severity.ERROR,
            "no baseline comparison found; evaluation must compare against explicit baselines",
            rule="M4C.baseline",
            suggestion="compare against at least one named baseline or benchmark",
        )
    ]


@register("methods.leakage", "Human-subject cross-validation avoids identity leakage", "M4C")
def methods_leakage(ctx: CheckContext) -> list[Finding]:
    """Pooled cross-validation leaks subject identity across folds."""
    body, offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    first = _first_phrase(lowered, vocab.CROSS_VALIDATION)
    if first is None:
        return []
    if any(term in lowered for term in vocab.SUBJECT_SAFE_VALIDATION):
        return []
    human_subjects = bool(ctx.config.get("venue.human_subjects")) or any(
        term in lowered for term in vocab.HUMAN_SUBJECT_TERMS
    )
    if not human_subjects:
        return []
    pos, phrase = first
    return [
        ctx.finding(
            "methods.leakage",
            Severity.WARNING,
            f"pooled {phrase!r} on human-subject data risks identity/feature leakage across folds",
            pos=offset + pos,
            rule="M4C.leakage",
            suggestion="use leave-one-subject-out or grouped validation so no subject spans folds",
        )
    ]


@register("methods.reproducibility", "Reproducibility details are reported", "M4C")
def methods_reproducibility(ctx: CheckContext) -> list[Finding]:
    """Reviewers must be able to rerun the study from the paper alone."""
    body, _offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.REPRODUCIBILITY_TERMS):
        return []
    return [
        ctx.finding(
            "methods.reproducibility",
            Severity.WARNING,
            "no reproducibility details found; report seeds, software versions, hyperparameters, or code availability",
            rule="M4C.reproducibility",
            suggestion="state random seeds, software versions, hyperparameters, and where the code is available",
        )
    ]


@register("methods.robustness", "Robustness analyses are reported", "M4C")
def methods_robustness(ctx: CheckContext) -> list[Finding]:
    """A single headline number hides how and when the method fails."""
    body, _offset = latex.document_body(ctx.doc.text)
    lowered = body.lower()
    if any(term in lowered for term in vocab.ROBUSTNESS_TERMS):
        return []
    return [
        ctx.finding(
            "methods.robustness",
            Severity.WARNING,
            "no robustness analysis found; add perturbation stress tests / ablations mapping failure thresholds",
            rule="M4C.robustness",
            suggestion="add ablations or perturbation stress tests that map where performance degrades",
        )
    ]


@register("methods.future-tense", "Methods are reported in past tense", "M4C")
def methods_future_tense(ctx: CheckContext) -> list[Finding]:
    """Completed work is reported in past tense, not proposal-style future tense."""
    section = _methods_section(ctx.doc.text)
    if section is None:
        return []
    lowered = section.body.lower()
    findings: list[Finding] = []
    for phrase in vocab.FUTURE_TENSE_METHODS:
        start = 0
        while True:
            pos = lowered.find(phrase, start)
            if pos < 0:
                break
            findings.append(
                ctx.finding(
                    "methods.future-tense",
                    Severity.WARNING,
                    f"future tense {phrase.strip()!r} in the methods section; completed work is reported in past tense",
                    pos=section.body_start + pos,
                    rule="M4C.future-tense",
                    suggestion="rewrite the procedure in past tense (e.g. 'we recorded', not 'we will record')",
                )
            )
            start = pos + 1
    findings.sort(key=lambda finding: (finding.path, finding.line))
    return findings
