#!/usr/bin/env python3
"""Test a montage hypothesis for the unlabeled SHAPE channels from the data alone.

Hypothesis (from Nelson et al., 2015, Psychol. Med.: BioSemi ActiveTwo, "thirty-four
electrodes ... based on the international 10/20 system", mastoid reference): channels
0-31 are the BioSemi 32-channel layout in its standard order, and channels 32-33 are two
added midline sites. Test: Mantel correlation between template inter-electrode distance
and the cohort's channel correlation (nearby electrodes correlate more under volume
conduction), against 2000 random channel orders. This is evidence, not a substitute for
the lab confirming the channel list.

    python experiments/check_shape_montage.py --shape-dir /path/batch_data
"""
from __future__ import annotations

import sys

import numpy as np

from common import base_parser, load_raw, write  # noqa: E402


def main() -> int:
    import mne
    args = base_parser(__doc__).parse_args()
    raw, y, g = load_raw(args)
    X = raw - raw.mean(axis=1, keepdims=True)
    m = X.reshape(-1, X.shape[2])
    C = m.T @ m
    R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    mont = mne.channels.make_standard_montage("biosemi32")
    names = mont.ch_names
    pos = np.array([mont.get_positions()["ch_pos"][n] for n in names])
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    iu = np.triu_indices(32, 1)
    rng = np.random.default_rng(0)

    def mantel(cols):
        Rs = R[np.ix_(cols, cols)]
        obs = float(np.corrcoef(D[iu], Rs[iu])[0, 1])
        null = np.array([np.corrcoef(D[iu], Rs[np.ix_(q, q)][iu])[0, 1]
                         for q in (rng.permutation(32) for _ in range(2000))])
        return {"r": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                "p": float((1 + (null <= obs).sum()) / 2001)}

    out = {"hypothesis": "channels 0-31 = BioSemi-32 standard order; 32-33 = extra midline sites",
           "biosemi32_order": names,
           "mantel_channels_0_31": mantel(list(range(32))),
           "mantel_shifted_2_33": mantel(list(range(2, 34))),
           "extra_channels_top_correlates": {
               str(k): [[names[t], float(R[k, t])] for t in np.argsort(-R[k, :32])[:3]] for k in (32, 33)},
           "most_negative_pair_r": float(np.min(R + 2 * np.eye(34)))}
    print(out)
    write(out, args.out or "shape_montage_check.json", args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
