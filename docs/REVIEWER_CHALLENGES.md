# Reviewer challenge matrix

Updated 2026-11-22. Each row is an objection a skeptical referee is likely to raise, the
evidence that answers it, and what is still open.

## Paper A (TNSRE)

| # | Objection | Evidence in hand | Open |
|---|---|---|---|
| A1 | "The identity is two lines of algebra. Subtracting an epoch mean obviously moves the response." | The contribution is the set, not the line: the general form with no baseline assumption (exact to 1.6e-15 on every epoch), the scale route (Proposition 2), the causal-encoder corollary, and the averaging law with a pre-committed prediction. The scale route is not obvious: scale-only normalization passes a first-order pre-stimulus check (0.349, p = 0.25) while a second-order feature reads 0.450 (X09) | none |
| A2 | "van Driel et al. (2021) already showed that detrending over the whole trial displaces decoding into the baseline. Epoch demeaning is order-zero detrending." | Now stated in the introduction. What that paper does not contain: a closed form, the dividing statistic, an averaging law, trained and spiking encoders, a codebase audit | Read the full van Driel text once more before submission to confirm that no order-zero or z-scoring case appears in it. Today's reading was the abstract and search excerpts |
| A3 | "Does it matter? Full-epoch accuracy hardly moves." | Stated openly: EEGNet full epoch +0.023 [-0.005, +0.052]. The consequence is for windows that include pre-stimulus samples and for latency claims, where the leak is the whole margin (reservoir 0.487 to 0.294). New subsection on online systems | A referee may still rank the paper as narrow. This is the main acceptance risk |
| A4 | "One private cohort." | Five public ERP CORE paradigms with a statistic registered in the dissertation repository before any ERP CORE result. X10 raises the permutation count to 2000 and adds Holm adjustment | Complete. N170 p = 0.0040, P3 0.0005, N400 0.0005, N2pc 0.0010, MMN 0.055. Holm-adjusted 0.008, 0.0025, 0.0025, 0.003, 0.055. MMN moved from 0.045 (200 permutations) to 0.055 (2000) |
| A5 | "p = 0.005 is the floor of your test, and five paradigms are uncorrected." | X10 | as A4 |
| A6 | "The averaging law misses P3." | Reported, with the reason: the k = 1 estimate is 0.504, where the probit transform amplifies sampling error, and the rare class limits k to 4. The theory section now separates the exact statement (D grows as root k) from the approximate one (the accuracy formula) | A per-participant test of the law would be stronger than the group curve. Not run |
| A7 | "70% unreported does not mean 70% affected." | The text now says so in those words | none |
| A8 | "The audit of ten codebases and 40 arXiv papers is a convenience sample." | Protocol committed before inspection, every item and deciding line listed | arXiv is not the published literature. A sample from the target journal itself would be more persuasive. Not run |
| A9 | "Why is a spiking reservoir in a normalization paper?" | It is the probe that isolates the corollary: a fixed causal system truncated at onset. EEGNet shows the same effect for a trained model | A referee may ask for it to be shortened |
| A10 | "MMN is significant unscaled as well." | Reported: 0.537 unscaled against 0.553 z-scored, and the text says its pre-stimulus interval is not a clean null | none |

## Paper B (venue pending)

| # | Objection | Evidence in hand | Open |
|---|---|---|---|
| B1 | SPL R2.1, R2.2: "How is centering applied to test subjects? This looks transductive." | It is, and the paper now says so. X11b on 211 participants: full panel 0.900, two of three conditions 0.820, single observation 0.679 | none |
| B2 | "Leave-one-out centering is your control." | It is not an independent control. Proposition 2(ii) shows it is K/(K-1) times full-panel centering | none |
| B3 | SPL R1.2, R2.9, R2.13: "Compare with Euclidean alignment and Riemannian methods." | X11 on SHAPE (211 participants): EA 0.834, panel centering 0.900, EA then centering 0.912. The prediction that EA would stay near the uncentered 0.679 was not supported on averages and held on single trials, which shows that part of the participant effect is a gain and not an offset | none. The tangent rows are from the third run, after two defects in the covariance handling were fixed (run record in the script) |
| B4 | SPL R1.4: "One dataset." | X11 on ERP CORE averages (five paradigms, all rise under panel centering). X12 on ERP CORE single trials: law error 0.004, break-even size correct in five of five | A second trial-averaged cohort with three or more conditions would match SHAPE more closely. The public IAPS dataset used in the dissertation's external replication is the candidate |
| B5 | "With a complete panel the task is a paired comparison. Of course it is easy." | Agreed, and it is the paper's point. Proposition 2(iii) gives the factor K/(K-1), and the joint-assignment readout (0.953 against 0.900) shows the design knowledge being used | The paper must never set 0.900 beside ordinary cross-subject accuracies |
| B6 | "Unequal trial counts per condition." | Proposition 1: a participant-specific class mix reintroduces a participant effect along the condition means | Not yet measured. P3 (20% rare) with resampled class mix is the natural test |
| B7 | SPL R2.4, R2.5: principal angles, interaction term | none | not run |
| B8 | SPL R1.7: ethics statement | none | needs the SHAPE protocol number from the data custodian |
| B9 | SPL R2.12: energy claim | removed from the plan | none |
| B10 | "Archived 70.5% does not reproduce." | Stated in X07: 67.5% from the committed code | The manuscript must use only numbers from committed runs |
| B11 | "A 1 to 4 point single-trial gain is small." | Stated in the abstract. The contribution is the rule for when centering helps and how many reference observations it needs, including the result that fewer than m* lowers accuracy | The strongest reply would be a paradigm with a large participant share on single trials. None is in hand |
| B12 | "Nine references." | none | The related-work section needs a proper literature pass: test-time adaptation, batch-normalization statistics per subject, mixed-effects decoding, the original repeated-measures normalization literature |
