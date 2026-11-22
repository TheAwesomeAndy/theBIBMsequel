# ARSPI-Net publication program: status dashboard

Updated 2026-11-22. Working branch: `journal-tnsre-final`. Companion files in `docs/`:
JOURNAL_MATRIX.md, REVIEWER_CHALLENGES.md, PROVENANCE_AUDIT.md, OPPORTUNITY_MAP.md.

## Dashboard

| | Paper A: normalization leakage | Paper B: reference centering |
|---|---|---|
| Location | `manuscript/` | `paperB/` |
| Central claim | Scaling an epoch by its own statistics writes post-stimulus information into the pre-stimulus interval by two exact routes, with a square-root averaging law for its size | Subject centering estimates a participant offset from m unlabeled reference observations. Separation is m/(m+1) of its asymptote, and centering helps only above a break-even size |
| Draft | Complete, 8 pages, builds clean | First full draft, 5 pages, builds clean |
| Target | IEEE TNSRE | IEEE TNSRE after Paper A, or J. Neural Eng. |
| Readiness | Not submission-ready. Science is in place. Four items below remain | Not submission-ready. Core evidence is in place. Related work and three reviewer analyses remain |
| Next action | Author review of the revised PDF | Literature pass for the related-work section |

## Paper A: what stands between the draft and submission
1. Author decisions: title (shortened this session), footnote and contact address. Length is settled: cut to exactly 8 pages on 2026-11-22 (text from about 5,430 to 4,560 words, three figures shortened, every result and number kept). The last page is full, so any addition needs an equal cut.
2. Co-author approval of the text and of the AI-use statement.
3. Crossref pass on six references added this session.
4. One full reading of van Driel et al. (2021) to confirm that it contains no order-zero or z-scoring case. The novelty statement in the introduction depends on it.

Scientific risk that no further experiment in hand removes: a referee may judge the
effect narrow, because full-epoch accuracy changes by +0.023 with an interval that
includes zero. The paper states this and argues from timing claims and causal
realizability.

## Paper B: what stands between the draft and submission
1. Related work: nine references today. Needs test-time adaptation, per-subject normalization layers, mixed-effects decoding, repeated-measures normalization.
2. Reviewer analyses not yet run: principal angles between participant and condition subspaces, the interaction term, the monotonicity test on the PCA sweep, a measured test of unequal class composition (Proposition 1 predicts the effect, P3 can test it).
3. Ethics statement for SHAPE, from the data custodian.
4. Author order and affiliations. The draft carries the names in the order of the earlier letter and no addresses.
5. A second trial-averaged cohort with three or more conditions would strengthen the panel result.

## Session record, 2026-11-22

**Verified**
- Paper A numbers against committed outputs (automated pass, zero unmatched) and the protocol by reading the code: fold-local PCA and scaler, within-participant permutations, label-free scaling.
- X09 reproduced the X01 anchors exactly (0.542, 0.346, 0.420, 0.297).
- X10 reproduced every X02 across-participant statistic to three decimals.

**Experiments completed** (all outputs committed)
- X09: location and scale routes, SHAPE and ERP CORE. All four pre-stated predictions held.
- X10: ERP CORE across-participant test with 2000 permutations and Holm adjustment.
- X11: alignment comparison, joint assignment, ERP CORE complete panels, and the panel controls on 211 participants (X11b).
- X12: reference-size law on single trials, five paradigms, with the summary script.

**Results that changed**
- MMN across-participant leak: p = 0.045 with 200 permutations, 0.055 with 2000. The paper already counted four of five paradigms, so no claim changes.
- N170: p = 0.0040, not the 0.005 floor reported before.
- SHAPE panel numbers for Paper B now refer to 211 participants (full panel 0.900, previously 0.912 on 212 with one all-zero epoch included).
- Euclidean alignment on SHAPE: predicted near 0.679, observed 0.834. Reported as an unsupported prediction and interpreted as a participant gain.

**Changed in the manuscripts**
- Paper A: general identity without a baseline assumption, Proposition 2 (scale route), stated assumptions of the averaging law, positioning against whole-trial detrending, new results subsection and table, online-systems subsection, revised reporting rule, codebase citations, shorter title, revised abstract (247 words).
- Paper B: created. Theory (two propositions, one corollary), methods, results, discussion, one figure, two tables.

**Runs that did not count**
- X10 first launch, stopped to change the permutation count for the unscaled control.
- X11 SHAPE, two earlier runs with defective covariance handling (run record in the script).
- X12 first run, stopped at N400 (run record in the script).
- ERP CORE cell `pre_logsd|pre_sd` in X09, constant by construction, annotated in the output.

**Uncertain**
- TNSRE acceptance figure and impact factor were found in the previous session and not re-read today.
- Venue metrics for J. Neural Eng. and TBME come from aggregators.
- Whether TNSRE's overlength rate matches its sister journals ($250 per page for the first two).

**Git state**
- Commits carry `Andrew Lane <>` with no email and no co-author trailer.
- Remote branches that still hold earlier identity metadata: `journal-tnsre`, `journal-tnsre-stonybrook`, `journal-tnsre-clean`, `journal-tnsre-noemail`. This session cannot delete remote branches. They can be deleted from the repository's Branches page. `journal-extension` and two `ccr-` branches also exist and were not inspected.

**Next highest-value action**
Author review of `manuscript/main.pdf`, then the Crossref pass and the van Driel reading,
which together make Paper A ready for co-author sign-off.
