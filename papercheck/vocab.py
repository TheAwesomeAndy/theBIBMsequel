# SPDX-License-Identifier: Apache-2.0
"""Shared vocabulary for all lexical checks.

Centralizing the word lists keeps every check module consistent and gives
users one place to see exactly what the harness looks for. Project-specific
additions belong in ``harness/forbidden_phrases.txt`` (scanned separately by
``scripts/check_forbidden_phrases.py``), not here.
"""
from __future__ import annotations

# --- Hype and overclaiming -------------------------------------------------

# Marketing language that top-tier venues treat as a red flag. Errors.
HYPE_ERROR = (
    "groundbreaking",
    "revolutionary",
    "paradigm shift",
    "paradigm-shifting",
    "game-changing",
    "game changing",
    "world-class",
    "best-in-class",
    "unparalleled",
    "unmatched performance",
    "remarkable breakthrough",
)

# Claims that need explicit evidence nearby; flagged as warnings.
HYPE_WARN = (
    "cutting-edge",
    "cutting edge",
    "unprecedented",
    "breakthrough",
    "for the first time",
    "superior performance",
    "outperforms all",
)

# Absolute claims. Science reports evidence, not certainty. Errors.
OVERCLAIM = (
    "proves that",
    "clearly proves",
    "definitively proves",
    "conclusively proves",
    "undoubtedly",
    "unquestionably",
    "it is obvious that",
    "obviously",
    "without a doubt",
    "irrefutable",
    "flawless",
    "guarantees that",
    "certainly demonstrates",
)

# Overclaim phrases with a concrete cautious replacement.
CAUTIOUS_REPLACEMENTS = {
    "this proves": "this suggests / this indicates",
    "these results prove": "these results suggest / these results indicate",
    "we prove that our": "we show that our / our results indicate that",
    "demonstrates conclusively": "provides evidence that",
    "confirms that": "is consistent with / supports the view that",
}

# Cautious interpretation verbs expected somewhere in the Discussion.
CAUTIOUS_VERBS = (
    "suggest",
    "suggests",
    "indicate",
    "indicates",
    "may ",
    "might ",
    "appears to",
    "appear to",
    "likely",
    "could ",
    "possibly",
    "seems to",
    "a plausible explanation",
    "one explanation",
)

# Speculative language that belongs in the Discussion, not the Results.
SPECULATION = (
    "might be due",
    "may be due",
    "may be explained",
    "possibly because",
    "perhaps",
    "we believe",
    "we speculate",
    "presumably",
    "one explanation",
    "a likely explanation",
    "this suggests that",
    "probably reflects",
)

# --- Structure and framing ---------------------------------------------------

# The forbidden roadmap paragraph ("celery, not spaghetti").
ROADMAP = (
    "the rest of this paper is organized as follows",
    "the rest of this paper is organised as follows",
    "the remainder of this paper is organized as follows",
    "the remainder of this paper is organised as follows",
    "the rest of the paper is organized as follows",
    "the rest of the paper is structured as follows",
    "the remainder of the paper is structured as follows",
    "this paper is organized as follows",
    "this paper is organised as follows",
    "this paper is structured as follows",
)

# Cliche openings that waste the page-1 hook.
CLICHE_OPENINGS = (
    "in recent years",
    "in recent decades",
    "with the rapid development",
    "with the rapid advancement",
    "with the advent of",
    "with the rise of",
    "nowadays",
    "in today's world",
    "in the modern era",
    "since the dawn of",
    "over the past few decades",
    "over the last few decades",
)

# Generic title fillers (checked as title prefixes).
TITLE_FILLER_PREFIXES = (
    "an investigation into",
    "an investigation of",
    "a study of",
    "a study on",
    "a study into",
    "an analysis of",
    "an approach to",
    "a novel approach",
    "some observations on",
    "notes on",
    "research on",
    "towards understanding",
    "an exploration of",
)

# Research-gap archetypes. A compound gap needs at least two archetypes.
GAP_ARCHETYPES: dict[str, tuple[str, ...]] = {
    "lack_of_research": (
        "remains unexplored",
        "remain unexplored",
        "has received little attention",
        "have received little attention",
        "few studies",
        "little research",
        "no prior work",
        "has not been studied",
        "have not been studied",
        "underexplored",
        "remains an open question",
        "remains an open problem",
        "little is known",
        "scarcity of research",
    ),
    "no_consensus": (
        "no consensus",
        "lack of consensus",
        "conflicting results",
        "conflicting findings",
        "contradictory findings",
        "contradictory results",
        "remains debated",
        "remains contested",
        "mixed evidence",
        "mixed results",
        "inconsistent findings",
        "inconsistent results",
    ),
    "method_limitation": (
        "existing methods",
        "existing approaches",
        "prior methods",
        "prior approaches",
        "previous studies are limited",
        "suffer from",
        "suffers from",
        "fail to generalize",
        "fails to generalize",
        "fail to generalise",
        "do not account for",
        "does not account for",
        "cannot handle",
        "limitations of existing",
        "limitations of previous",
        "limitations of prior",
    ),
    "practical_bottleneck": (
        "in practice",
        "real-world",
        "real world",
        "deployment",
        "clinical practice",
        "practical settings",
        "practical applications",
        "in the field",
        "operational settings",
        "at scale",
    ),
}

# --- Related work tone -------------------------------------------------------

# Hostile take-downs of prior work ("credit is like love" violation).
HOSTILE = (
    "naive",
    "trivial",
    "inferior",
    "fails to",
    "failed to",
    "poorly designed",
    "deeply flawed",
    "fundamentally flawed",
    "useless",
    "incorrectly claims",
    "incorrectly claim",
    "so-called",
    "merely",
    "simply wrong",
    "misguided",
)

# Generous adjectives to suggest instead.
GENEROUS = ("insightful", "inspiring", "seminal", "elegant", "influential", "pioneering", "fascinating")

# --- Machine-writing tells ---------------------------------------------------

AI_TELLS = (
    "delve into",
    "delves into",
    "delving into",
    "it is worth noting that",
    "it should be noted that",
    "it is important to note that",
    "plays a crucial role",
    "plays a vital role",
    "plays a pivotal role",
    "in the realm of",
    "a testament to",
    "underscores the importance",
    "rich tapestry",
    "multifaceted landscape",
    "leverage the power of",
    "unlock the potential",
    "seamlessly integrates",
    "in the ever-evolving",
)

CONTRACTIONS = (
    "don't", "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "it's", "we've", "we're", "they're", "let's", "that's", "there's",
)

WEASEL = ("very ", "really ", "extremely ", "hugely ", "incredibly ", "a lot of ", "massive ")

# --- Methodology rigor ---------------------------------------------------------

CROSS_VALIDATION = (
    "k-fold", "kfold", "k fold", "10-fold", "ten-fold", "5-fold", "five-fold",
    "cross-validation", "cross validation",
)

SUBJECT_SAFE_VALIDATION = (
    "leave-one-subject-out",
    "leave one subject out",
    "loso",
    "leave-one-participant-out",
    "subject-independent",
    "subject independent",
    "subject-wise",
    "cross-subject",
    "held-out subjects",
    "held-out participants",
    "grouped by subject",
    "groupkfold",
    "group k-fold",
)

HUMAN_SUBJECT_TERMS = ("participants", "subjects", "patients", "volunteers", "human data", "human subjects")

ETHICS_TERMS = (
    "ethics",
    "ethical approval",
    "irb",
    "institutional review board",
    "informed consent",
    "declaration of helsinki",
    "ethics committee",
    "review board approval",
)

REPRODUCIBILITY_TERMS = (
    "random seed",
    "seed",
    "software version",
    "version ",
    "hyperparameter",
    "hyper-parameter",
    "implementation detail",
    "code is available",
    "code are available",
    "publicly available",
    "open-source",
    "open source",
)

ROBUSTNESS_TERMS = (
    "ablation",
    "noise injection",
    "perturbation",
    "jitter",
    "channel dropout",
    "stress test",
    "stress-test",
    "sensitivity analysis",
    "robustness",
    "failure threshold",
    "degradation",
)

BASELINE_TERMS = ("baseline", "baselines", "compared against", "compared with", "comparison with", "benchmark")

FUTURE_TENSE_METHODS = ("we will ", "we are going to ", "we plan to measure", "will be measured in this study")

# --- Limitations, availability, ethics statements ------------------------------

LIMITATION_TERMS = ("limitation", "limitations", "caveat", "caveats", "boundary condition", "threats to validity")

PIVOT_TERMS = (
    "future work",
    "future research",
    "future studies",
    "nevertheless",
    "nonetheless",
    "despite this",
    "despite these",
    "we plan to",
    "could be addressed",
    "can be addressed",
    "could be mitigated",
    "can be mitigated",
    "remains robust",
    "does not affect the main conclusion",
)

DATA_AVAILABILITY_TERMS = (
    "data availability",
    "data are available",
    "data is available",
    "dataset is available",
    "datasets are available",
    "code is available",
    "code and data",
    "available at",
    "upon reasonable request",
    "supplementary material",
)

# --- Blinding -------------------------------------------------------------------

SELF_REVEAL = (
    "our previous work",
    "our prior work",
    "our earlier work",
    "in our previous paper",
    "as we showed in",
    "extends our",
)

# Repository hosts that leak identity in double-blind review. The
# anonymized hosts below are explicitly allowed.
IDENTIFIABLE_HOSTS = ("github.com/", "gitlab.com/", "bitbucket.org/", "osf.io/", "zenodo.org/record")
ANONYMIZED_HOSTS = ("anonymous.4open.science", "anonymized", "anonymous.for.review")

# --- Rebuttal tone -----------------------------------------------------------------

EMOTIONAL_REBUTTAL = (
    "unfair",
    "insulting",
    "offensive",
    "clearly did not read",
    "did not bother",
    "nonsense",
    "absurd",
    "ridiculous",
    "outrageous",
    "incompetent",
    "we strongly disagree",
    "completely wrong",
    "utterly",
    "lazy review",
)

POLITENESS_MARKERS = ("thank", "we appreciate", "we are grateful", "insightful", "helpful")

# --- Figures ------------------------------------------------------------------------

COLOR_WORDS = ("red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta", "pink", "brown", "gray", "grey")
COLOR_TARGETS = ("line", "lines", "curve", "curves", "bar", "bars", "dot", "dots", "point", "points", "region", "area", "marker", "markers", "trace", "traces")
