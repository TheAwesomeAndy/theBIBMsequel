# Theory notes for the journal paper

Propositions 1 and 2 are the BIBM camera-ready statements (`bibm_baseline/manuscript/main_bibm2026.tex`,
Sec. II-E). Proposition 3 and the fill-bias index are new for the journal paper. Every
statement below has a synthetic test in `tests/test_core.py`.

## Setting

A fixed encoder maps a trial to features `x in R^D`, grouped in per-electrode blocks. The
readout standardizes with training statistics, `z = D_sigma^{-1}(x - mu)`, and computes class
logits `l = W z + b` (multinomial logistic regression; the binary case uses logits `(0, w.z + b)`).
`S` is the set of coordinates of the dropped electrodes.

## Proposition 1 (BIBM): fill rule and re-centering

Zero fill (`x_S = 0`) and mean fill (`x_S = mu_S`) differ by the fixed shift
`z_S^zero - z_S^mean = -D_{sigma,S}^{-1} mu_S`. Under a constant re-centering `x -> x - c`,
the clean input and the mean-fill input are unchanged, while the zero-fill input shifts by
`D_{sigma,S}^{-1} c_S`. The two fills coincide iff `mu_S = 0`.

Test: `test_proposition1_mean_fill_invariant_zero_fill_not`.

**Corollary (logit shift).** For a linear readout the zero-fill logits equal the mean-fill
logits plus the constant vector `delta = -W_{:,S} (mu_S / sigma_S)`, the same for every test
observation. Softmax ignores a common offset, so only `delta - mean(delta)` matters.

Test: `test_zero_fill_shift_formula_exact` (exact to 1e-9).

## Proposition 2 (BIBM): chance floor of raw retention

If `E[BA_pert] >= c`, then `E[R] >= c / BA_clean`, which tends to 1 as `BA_clean -> c`, while
above-chance retention `A` is 0 at chance. To first order the sampling error of `A` is
`s / (BA_clean - c)`.

## Proposition 3 (new): zero fill drives a non-centered encoder to chance

Consider the family of encoders `x_t = x + t a` for a fixed direction `a` and scalar `t`
(the same encoder with its origin moved). Fit the readout on each.

1. The standardized training data, and therefore `W` and `b`, do not depend on `t`
   (`mu_t = mu + t a`, `sigma_t = sigma`). By Proposition 1 the mean-fill logits of every
   test observation, and hence mean-fill balanced accuracy, do not depend on `t`.
2. The zero-fill logit shift is `delta_t = delta_0 + t u` with `u = -W_{:,S} (a_S / sigma_S)`.
   Suppose `u` has a unique largest coordinate `k*`. For a test observation with mean-fill
   logits `m`, zero fill predicts `k*` as soon as
   `t > max_{k != k*} (m_k + delta_{0k} - m_{k*} - delta_{0k*}) / (u_{k*} - u_k)`.
   A test set is finite, so beyond the largest such threshold every observation is
   assigned to `k*`: recall is 1 for class `k*` and 0 for the others, and balanced accuracy
   equals `1/K`.

So the measured zero-fill robustness of an encoder can be moved anywhere between its
mean-fill value and chance by a change of origin that carries no information. Test:
`test_proposition3_zero_fill_tends_to_chance`.

Remarks for the paper:
* The result needs only a linear readout on standardized features, so it covers every
  fixed encoder in the study and the final layer of a trained network whose dropped
  inputs are zero-padded.
* `t -> -infinity` drives predictions to the class with the smallest coordinate of `u`.
* kNN fill is translation-equivariant, so it is also invariant to re-centering.

## Fill-bias index (new, empirical)

`B = spread(delta - mean(delta)) / median top-two logit margin on the training data.`

`B` uses no test labels. Hypothesis for the journal paper: across encoders, dropout rates
and datasets, the observed loss from zero fill relative to mean fill grows with `B`.
This is a prediction to be tested (Phase 4 in `docs/ROADMAP.md`), not a theorem. Every
experiment logs `B` per encoder and condition (`experiments/common.py`, `TheoryLog`).

## Reservoir: one-step branching count

From BIBM: a flipped spike in unit `j` changes unit `i`'s next input by `W_ij` and flips it
only if its pre-threshold membrane is within `|W_ij|` of `theta`, so one flip produces about
`m(rho) = rho * p_theta * kappa_bar` further flips, with `kappa_bar` the mean column l1 norm
of `W` at unit spectral radius (13.4 for seed 42, checked in
`test_membrane_density_and_branching`). BIBM left `p_theta` unmeasured;
`LIFReservoir.encode(..., membrane_stats=True)` now measures it, and
`experiments/run_reservoir_dynamics.py` reports `rho_c` where `m = 1` for every weight seed.

Note found while writing the tests: with `rho = 0` a flip cannot reach other units, but it
changes the flipped unit's own reset, so damage is confined to one unit (at most
`1/N_RES`), not exactly zero (`test_damage_zero_without_recurrence`).
