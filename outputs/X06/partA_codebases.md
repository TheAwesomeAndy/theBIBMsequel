# X06 Part A: codebases (inspected 2026-11-21, shallow clones of default branches)

Codes follow experiments/X06/PROTOCOL.md. Repositories were read, not run.

| Codebase | Code | Deciding source | Note |
|---|---|---|---|
| TorchEEG | OWN | `torcheeg/transforms/numpy/normalize.py`, `MeanStdNormalize.apply`: with no mean/std given, `mean = eeg.mean(); std = eeg.std()` of the sample itself; default `axis=None` | Used this way in `examples/examples_dreamer_tsception.py` (`transforms.MeanStdNormalize()`) |
| BENDR (via DN3) | OWN | DN3 `dn3/transforms/instance.py`, `MappingDeep1010`: "normalizes data between [-1, 1]" with `min_max_normalize` over each trial (`dn3/utils.py`) | Also appends a channel with the trial's relative range |
| BIOT | OWN (scale only) | `utils.py`, dataset `__getitem__`: `X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)` | Divides by each sample's own 95th percentile; no mean subtraction, so only the denominator route of Sec. III applies |
| braindecode | TRAIN | `braindecode/preprocessing/preprocess.py`, `exponential_moving_standardize` applied to continuous recordings before windowing in `examples/model_building/plot_bcic_iv_2a_moabb_trial.py` | Causal running statistics, past samples only |
| EEG Conformer | TRAIN | `conformer.py`: `target_mean = np.mean(self.allData); target_std = np.std(self.allData)` on training data, applied to test data | |
| EEGNet reference (ARL eegmodels) | FIXED | `examples/ERP.py`: "scale by 1000 due to scaling sensitivity in deep learning" | |
| LaBraM | FIXED | `engine_for_finetuning.py`: `samples = samples.float().to(device) / 100` | |
| CBraMod | FIXED | `datasets/*_dataset.py`: `return data/100, label` | |
| EEGPT | FIXED | `downstream/linear_probe_EEGPT_KaggleERN.py`: `x = x/10` then `x - x.mean(dim=-2)` (mean over channels, a spatial re-reference) | Its BENDR baseline scripts use per-trial min-max (`downstream/utils.py`, `min_max_normalize`) |
| MOABB | NONE | `pipelines/*.yml`: Xdawn, covariance, CSP pipelines with no amplitude scaling step | |

Counts: OWN 3 (one scale-only), TRAIN 2, FIXED 4, NONE 1, of 10.
