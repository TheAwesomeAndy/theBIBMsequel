#!/usr/bin/env python3
"""Per-unit beat tagging: for each of the 256 LIF units, Welch tagging z at k*f_beat
(same neighbour-baseline statistic as v3). Reports the fraction of units with z>1.96
(chance ~0.025) and the best units, for real drive and phase-randomised drive, group PC1
and 5 participants.  Also: tagging z of the smoothed spike *count* summed over units
(= population rate) and of the first PC of unit rates (unsupervised linear readout)."""
import sys, os, json, numpy as np
from scipy.signal import resample_poly
from nmedt_pilot import load_song, pc1_component, NOMINAL_BPM
from nmedt_pilot_v3 import refine_beat_freq, tag_z, phase_randomize
sys.path.insert(0, '/home/user/dissoAdventureExperiments/chapter5Experiments')
from experiment_zero import LIFReservoir
mat = sys.argv[1]; song = int(os.path.basename(mat).split('_')[0].replace('song', ''))
data, fs = load_song(mat); pc1, w = pc1_component(data)
f_beat = refine_beat_freq(pc1[int(10*fs):], fs, NOMINAL_BPM[song]/60)['f_beat']
ks = (1, 2, 4, 8); rng = np.random.RandomState(42); res = LIFReservoir(seed=42)
def analyse(name, sig):
    x = resample_poly(sig[int(10*fs):].astype(float), 256, int(fs)); x = (x-x.mean())/x.std()
    for lab, xx in (('real', x), ('phase-rand', phase_randomize(x, rng))):
        S = res.forward(xx)
        zu = np.array([tag_z(S[:, j], 256.0, f_beat, ks) for j in range(S.shape[1])])   # 256 x 4
        frac = (zu > 1.96).mean(0); best = zu.max(0)
        U, s, Vt = np.linalg.svd(S - S.mean(0), full_matrices=False); zpc = tag_z(U[:, 0]*s[0], 256.0, f_beat, ks)
        zpop = tag_z(S.mean(1), 256.0, f_beat, ks)
        print(f"[{name:>8} {lab:>10}] units z>1.96 frac@(1,2,4,8)f={np.round(frac,3)} (chance .025) | max unit z={np.round(best,1)} | pop-rate z={np.round(zpop,1)} | unit-PC1 z={np.round(zpc,1)}")
    print(f"[{name:>8}      drive] z={np.round(tag_z(x,256.0,f_beat,ks),1)}")
analyse('groupPC1', pc1)
for p in range(5):
    analyse(f'P{p+1:02d}', w @ (data[:, :, p] - data[:, :, p].mean(1, keepdims=True)))
