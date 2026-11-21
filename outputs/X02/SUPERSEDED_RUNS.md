# Superseded X02 runs (kept for provenance)

## Run 1: pre-stimulus mean feature (degenerate after baseline correction)
```
N170  none     across 0.463 (p=0.738)  within 0.510 [0.490,0.530] p=0.327  38s
N170  pre_sd   across 0.452 (p=0.735)  within 0.507 [0.486,0.528] p=0.519  65s
N170  epoch_z  across 0.603 (p=0.022)  within 0.531 [0.506,0.555] p=0.0148  99s
N170  post_z   across 0.588 (p=0.031)  within 0.532 [0.508,0.557] p=0.0106  131s
P3    none     across 0.643 (p=0.018)  within 0.502 [0.495,0.509] p=0.554  172s
P3    pre_sd   across 0.560 (p=0.192)  within 0.498 [0.492,0.504] p=0.424  199s
P3    epoch_z  across 0.630 (p=0.010)  within 0.520 [0.507,0.534] p=0.00437  235s
P3    post_z   across 0.635 (p=0.013)  within 0.516 [0.504,0.529] p=0.0121  269s
```

## Run 2: four pre-stimulus bin means, 112 features, single readout
```
N170  none     across 0.502 (p=0.465)  within 0.510 [0.490,0.531] p=0.32  58s
N170  pre_sd   across 0.500 (p=0.529)  within 0.504 [0.483,0.526] p=0.685  96s
N170  epoch_z  across 0.600 (p=0.062)  within 0.528 [0.507,0.548] p=0.0102  134s
N170  post_z   across 0.608 (p=0.035)  within 0.530 [0.510,0.550] p=0.0042  177s
P3    none     across 0.525 (p=0.413)  within 0.498 [0.489,0.508] p=0.736  229s
P3    pre_sd   across 0.590 (p=0.078)  within 0.492 [0.484,0.501] p=0.0649  269s
P3    epoch_z  across 0.583 (p=0.099)  within 0.524 [0.508,0.539] p=0.0036  310s
```

Run 2 departed from the statistic pre-registered in TheAwesomeAndy/dissoAdventuresv2 (experiments/zscore_artifact/MECHANISM.md and run_erpcore.py, commit 69b6c97): time-resolved decoding in 4-sample bins with the mean accuracy over the pre-stimulus bins as the test statistic. The final X02 adopts that statistic. Both runs above were stopped part-way and are not reported as results.
