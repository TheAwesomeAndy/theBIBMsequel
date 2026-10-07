# Rigor policy

Methodological defenses that the harness expects.

- Baselines: compare against contemporary architectures, not only
  classical models (`methods.baseline`).
- Leakage: for human-subject data use subject-wise validation
  (leave-one-subject-out), never pooled k-fold; defend against identity
  and feature leakage explicitly (`methods.leakage`).
- Robustness: inject noise (jitter, channel dropout) and run ablations
  that map failure thresholds (`methods.robustness`).
- Reproducibility: report random seeds, software versions,
  hyperparameters, and code availability (`methods.reproducibility`).
- Statistics: "significant" implies a reported test; give the p-value,
  effect size, or interval, or use "substantial" (`lang.significant`).
  No p-hacking, no selective reporting.
- Tense: report completed work in the past tense (`methods.future-tense`).
