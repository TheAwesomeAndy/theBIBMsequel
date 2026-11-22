# Dissertation publication opportunity map

Updated 2026-11-22. Standing of each dissertation idea is taken from
`dissoAdventuresv2/docs/RESEARCH_MAP.md` (read this session, not re-run). "Leak status"
says whether the result is known to depend on per-epoch z-scored input.

## Already published
| Work | Venue | Note |
|---|---|---|
| ARSPI-Net framework papers | LISAT 2023, LISAT 2024 | cited in Paper A |
| Reservoir as a perturbation-characterized measurement operator | IEEE BIBM 2026 | frozen. Its numbers reproduce. Paper A changes the interpretation of the 0.487 decoding figure only |

## Claimed by the two active papers
| Paper | Central claim | Dissertation material used |
|---|---|---|
| A | Epoch-own scaling writes post-stimulus information into the pre-stimulus interval, by two exact routes, with a square-root averaging law | Research-map idea 10 (ZS-01, ZS-02). The BIBM reservoir is the case study |
| B | Subject centering is an estimate of a participant offset from unlabeled reference observations, and its gain follows a reference-size law | Research-map idea 8 (subject variance dominance), the Chapter 5 centering result |

## Existing drafts outside this repository
| Draft | State known to this session | Action before submission |
|---|---|---|
| Channel-loss fill-rule paper ("The Weight of Nothing") | Draft exists, earlier planned for J. Neural Eng. | Not checked for z-scored input. The BIBM propositions it builds on are scoring-rule statements and hold for any input, but every magnitude must be re-measured under leak-free scaling |
| Rhythmic entrainment (music EEG) | Single-author manuscript in its own repository | Not read this session. Check its input scaling against Proposition 1 and 2 of Paper A |

## Unclaimed results with publication potential
| Candidate | Evidence today | What it needs | Realistic venue type |
|---|---|---|---|
| C. What a spiking reservoir adds to ERP decoding, measured against bin-matched amplitude controls | Research-map ideas 4, 5, 6: accuracy flat across recurrent gain 0 to 0.9, no echo-state convergence at gain 0.9, bin-matched amplitudes equal the reservoir code. Committed v2 records | One more public dataset and a leak-free rerun. The claim is a control methodology for reservoir papers, with the ARSPI-Net numbers as the case | A neuromorphic or neural-engineering journal. This is the natural home of the "second BIBM journal paper" |
| D. Temporal order and electrode identity carry the decodable signal | Idea 7, confirmed with corrected controls | Too small alone. Fold into C | none alone |
| E. External replication on the public IAPS dataset (DataverseNL) | Idea 16, exploratory, inherits a known handicap | Becomes the second dataset of C or of B | supports C or B |

## Not publishable as they stand
| Idea | Standing | Why |
|---|---|---|
| Structure-function coupling kappa | Archived, null invalid | Needs a valid null before any claim |
| Dynamical descriptors, Lyapunov exponent | Archived | Computed on z-scored input with the v1 protocol |
| Graph propagation regime boundary | Archived | Not re-tested under the corrected protocol |
| Clinical correlates (SUD, PTSD) | Not detected under a corrected family | No positive finding to report |
| Four-level interpretability statistic r = 0.82 | Withdrawal recommended | The statistic was found invalid |
| Random projection beats PCA | Withdrawn | |
| Energy efficiency | Never measured | The SPL reviewers already objected to this claim |

## Order of work
1. Paper A to submission.
2. Paper B to a full draft.
3. Candidate C, only after 1 and 2. It reuses the X01 and X09 infrastructure and the
   committed v2 records, and it does not overlap A or B.

Three papers with distinct claims are supportable from the evidence. A fourth would need
a new experiment that does not exist yet.
