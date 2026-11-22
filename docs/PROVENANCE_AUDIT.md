# Experiment and provenance audit

Updated 2026-11-22. Every central claim, its script, its output, its test, and where it
appears. "Verified today" means the number in the manuscript was compared with the
committed JSON in this session. "Carried" means it was verified in the 2026-11-21
session by the automated number audit and not re-derived today.

## Paper A

| Claim | Script | Output | Test | Manuscript | Status |
|---|---|---|---|---|---|
| General identity, no baseline assumption, exact | `experiments/X09/run.py` | `outputs/X09/x09_shape.json`, `x09_erpcore.json` | max abs error 1.6e-15 (SHAPE), 1.8e-15 (ERP CORE) | Prop. 1, Sec. IV-H | new, verified today |
| Baseline-corrected identity on SHAPE, r = 0.9965 | `experiments/X01/identity.py` | `outputs/X01/x01c_identity.json` | correlation, agreement of predictions | Sec. IV-A, Fig. 1 | carried |
| Pre-stimulus mean 0.542 against 0.346 | `experiments/X01/run.py` | `outputs/X01/x01_results.json` | participant bootstrap, 200 within-participant permutations | Table I | carried. Reproduced again today inside X09 (0.542, 0.346) |
| Reservoir 0.487 to 0.294, truncated reservoir 0.420 to 0.297 | `experiments/X01/run.py`, `null_reservoir.py` | `x01_results.json`, `x01b_reservoir_null.json` | bootstrap, 50 permutations with fold-local PCA | Table I, Sec. IV-C | carried. Truncated-reservoir values reproduced today in X09 (0.420, 0.297) |
| Independent implementation agrees | BIBM repository, branch `ccr-868188eb-17db7r` | that repository | three-decimal agreement | abstract, Sec. IV-C | carried. Not re-read today |
| EEGNet 0.539 against 0.300 | `experiments/X04/run.py` | `outputs/X04/x04_results.json` | bootstrap | Table I, Sec. IV-D | carried |
| ERP CORE across-participant leak, five paradigms | `experiments/X02/run.py` | `outputs/X02/x02_results.json` | 200 permutations | Table II, Fig. 3 | carried |
| The same with 2000 permutations and Holm adjustment | `experiments/X10/run.py` | `outputs/X10/x10_results.json` | observed statistic equals X02 to three decimals in every completed cell | Table II (marked values) | new, complete |
| Averaging law | `experiments/X03/run.py` | `outputs/X03/x03_results.json` | prediction committed before the run | Sec. IV-F, Fig. 4 | carried |
| Location and scale routes | `experiments/X09/run.py` | `x09_shape.json`, `x09_erpcore.json` | bootstrap, 200 permutations (SHAPE), 500 (ERP CORE), four predictions written in the script before the run | Prop. 2, Table III, Sec. IV-H | new, verified today |
| Prevalence: 3 of 10 codebases, 28 of 40 papers | `experiments/X06/PROTOCOL.md` | `outputs/X06/*` | protocol committed before inspection | Sec. IV-G | carried |

Checks made on the code today, by reading it:
- Scaling is applied per epoch and channel before any fold is formed, and uses no label.
- `StandardScaler` and the block PCA are fitted on training rows inside each fold (`sequel/common.py`, `oof_proba`, `_block_pca`).
- Permutations shuffle labels within a participant, which keeps the repeated-measures structure (`label_permutation_null`, X02 `main`).
- ERP CORE permutation nulls use one cross-validation repeat and the observed statistic uses five. The null is therefore wider than the statistic's own sampling distribution, which makes the test conservative.
- Three superseded or degenerate computations are recorded and excluded: the two stopped X02 designs (`outputs/X02/SUPERSEDED_RUNS.md`) and the constant feature `pre_logsd|pre_sd` on ERP CORE (X09 run record).

Open items for this paper:
- Eight references were added today (BIOT, BENDR, TorchEEG, LaBraM, CBraMod, EEGPT, MOABB, AdaBrain-Bench). TorchEEG and AdaBrain-Bench were checked against a source today. The other six are from recollection and need the same Crossref pass the first 37 received.
- The committed code names the excluded SHAPE participant by number (`sequel/common.py`, `EXCLUDE`). A participant number in a public repository is a question for the data custodian.

## Paper B

| Claim | Script | Output | Test | Status |
|---|---|---|---|---|
| Centering gain persists under leak-free scaling | `experiments/X07/run.py` | `outputs/X07/x07_results.json` | participant bootstrap | carried |
| Panel-size controls, 212 participants | `experiments/X08/run.py` | `outputs/X08/x08_results.json` | 50 permutations | carried |
| Panel-size controls, 211 participants | `experiments/X11/panels.py` | `outputs/X11/x11_panels.json` | 100 permutations | new, complete |
| Alignment comparison and joint assignment | `experiments/X11/run.py shape` | `outputs/X11/x11_shape.json` | 4000 participant bootstraps | new, third run. The two earlier runs are described in the script's run record |
| Complete panels on ERP CORE averages | `experiments/X11/run.py erpcore` | `outputs/X11/x11_erpcore.json` | 500 permutations | new |
| Reference-size law and break-even size | `experiments/X12/run.py`, `summarize.py` | `outputs/X12/x12_results.json`, `x12_law_summary.json` | law written in the script before the run. Mean absolute error 0.004 over 30 points | new, second run. The first was discarded after N170 and P3 because N400 has no participant with 128 trials. Its two paradigms were seen before the rerun and gave the same picture |

Prediction that was not supported: Euclidean alignment was predicted to stay near the
uncentered accuracy on SHAPE. It reached 0.834 against 0.679. On single trials the
prediction held.

## Code revision of the new outputs

The X09 to X12 scripts were written and run in one session and committed together in
`e99d42d`. Outputs that record a `git_commit` therefore show the parent commit
`af43a15`. Each final output was produced by the script version committed in `e99d42d`,
with two exceptions that the run records describe: the ERP CORE half of X09 ran before
the constant-feature guard was added (the affected cell is annotated in the output),
and the ERP CORE half of X11 ran before the covariance fixes, which touch only the
SHAPE half. A rerun from the committed scripts reproduces every number.
