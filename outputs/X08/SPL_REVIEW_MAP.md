# SPL resubmission: reviewer points mapped to evidence (2026-11-21)

Source of the rejected letter's numbers: `dissoAdventureExperiments/chapter5Experiments/experiment_zero.py`
(listed in that repo's REPRODUCTION_MAP as Tier 3, archived and non-prospective).
New evidence: theBIBMsequel `outputs/X07`, `outputs/X08`.

## Answered by new experiments

| Point | Reviewer request | Evidence | Status |
|---|---|---|---|
| R2.1, R2.2 | How is centering applied to test subjects? Controls: leave-one-out, training-only statistics, unequal condition counts | X08. Full panel 0.912, leave-one-out 0.917 (exactly 3/2 times full-panel centering), 2 of 3 conditions 0.821, single sample with no subject statistics 0.678; within-subject permutation null for full panel: mean 0.329, max 0.379 over 50 permutations (uncentered model; a centered-trained model gives 0.502) | Answered. Reframe as repeated-measures (transductive) normalization: the gain needs a multi-condition panel from the test subject |
| R2.7 | z-scoring; PCA fitted only on training folds? | X07. Centering gain +0.203 under per-epoch z, +0.219 under pre-stimulus scaling; raw EEG centered 0.877 (z) and 0.912 (unscaled). Reservoir with fold-local PCA (seeded rerun): gain +0.189 [0.154, 0.225] (z), +0.160 [0.126, 0.195] (pre-stim. scaling); with global PCA +0.203 (z), +0.156 (pre-stim.) | Answered; use unscaled or pre-stimulus scaling and fold-local PCA in the resubmission |
| R2.6 | Paired tests, CIs on gains | X07 participant-bootstrap CIs for every gain | Partly answered; add CI on rho and effect sizes |
| R1.6, R2.14 | Reproducibility | Public code (theBIBMsequel, arspiNetCenteringPaper) | Answered once the resubmission cites the repository |

## Need new work

| Point | Request | Proposed answer |
|---|---|---|
| R1.1 | Derivation steps, symbol definitions | Write the full derivation from the additive model to the centered contrast, including the 3/2 leave-one-out identity |
| R1.2, R2.9, R2.13 | Literature and alternatives: EA, Riemannian alignment, covariance harmonization, single-trial normalization | Cite the three suggested works plus He & Wu (EA, TBME 2020); run EA and subject-wise z-scoring as alternatives under the same panel rules |
| R1.3 | Monotonicity test for PCA regression | Jonckheere-Terpstra or Spearman trend test on the k sweep |
| R2.4 | Shared-subspace claim | Principal angles between subject and condition subspaces, fold-wise, with permutation null |
| R2.5 | Subject x condition interaction | Extend the variance decomposition with an interaction term (one observation per cell, so interaction equals residual; state this) |
| R1.4 | Generalizability beyond SHAPE | ERP CORE, already local: participant-level centering on condition averages, five paradigms |
| R1.7 | Ethics statement | IRB approval and consent statement from the SHAPE protocol (Nelson) |
| R2.10 | Clinical covariates in subject variance | Relate subject offsets to diagnosis, age, sex (exploratory) |
| R2.12 | Energy-efficiency claim | Remove |
| R2.11 | Overstated wording | Rewrite claims to match the evidence |
| R1.8 | Why gains differ across architectures | Relate gain to each representation's subject-variance fraction |
