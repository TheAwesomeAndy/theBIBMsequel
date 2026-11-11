"""Timing constants and window definitions.

Raw SHAPE epochs: 1229 samples at 1024 Hz, 34 channels, baseline-corrected over the first
205 samples, so stimulus onset is raw sample 205 (BIBM camera-ready audit, item A18).

Two analysed-signal conventions exist:

* ``legacy``    the BIBM signal ``X_ds = raw[0:1024:4]``: no anti-alias filter, 256 samples
                at 256 Hz, onset at sample 51 (raw 204 -> -0.98 ms), spans about -200..+800 ms.
* ``corrected`` FIR anti-alias decimation of ``raw[1:]`` by 4: 307 samples at 256 Hz,
                onset exactly at sample 51, spans -199..+996 ms.

All new windows are stated in milliseconds relative to stimulus onset. The BIBM sample-index
windows are kept verbatim (``LEGACY_*``) so that the BIBM numbers can be reproduced exactly.
"""
from __future__ import annotations

FS_RAW = 1024
ONSET_RAW = 205
DECIM = 4
FS = FS_RAW // DECIM          # 256 Hz
ONSET = 51                    # onset sample in both analysed conventions

# --- reservoir defaults (identical to BIBM) -------------------------------------------------
N_RES = 256
BETA = 0.05
THETA = 0.5
RHO = 0.9
WEIGHT_SEED = 42
PCA_K = 64

# --- validation protocol (identical to BIBM) ------------------------------------------------
CV_SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
N_BOOT = 4000
READOUT_C = 1.0

# --- BIBM legacy windows (sample indices on the legacy 256-sample signal) -------------------
LEGACY_RES_WINDOW = (10, 70)                                  # about -160..+73 ms
LEGACY_ERP_WINDOWS_MS = [(80, 200), (250, 450), (450, 800)]   # ms from FIRST sample
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

# --- corrected windows, ms relative to onset -------------------------------------------------
ERP_WINDOWS_REL_MS = [(-120, 0), (50, 250), (250, 600)]       # what BIBM meant, now exact
RES_WINDOWS_REL_MS = {
    "bibm": (-160, 74),      # the BIBM window, re-expressed (mostly pre-stimulus)
    "early": (0, 250),       # early components
    "late": (250, 800),      # P300 / late positive potential
    "post": (0, 800),        # full post-stimulus
}


def ms_to_sample(ms: float, onset: int = ONSET, fs: int = FS) -> int:
    """Sample index of a time in ms relative to onset (rounded)."""
    return int(round(onset + ms / 1000.0 * fs))


def window_samples(rel_ms: tuple[float, float], onset: int = ONSET, fs: int = FS) -> tuple[int, int]:
    return ms_to_sample(rel_ms[0], onset, fs), ms_to_sample(rel_ms[1], onset, fs)
