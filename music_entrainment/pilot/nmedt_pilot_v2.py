#!/usr/bin/env python3
"""NMED-T pilot v2: harmonic beat-locking of reservoir output vs its own drive.

Statistic (per drive, per harmonic k in {1,2,4}):
    L_k = | sum_t (r_t - mean r) exp(-i k phi_t) | / T
with phi_t the nominal beat phase (origin arbitrary).  Compared with a
circular-shift null (shift >= 10 s) -> z-score.  Same statistic applied to the
input drive x_t, so the reservoir's effect on detectability is a paired contrast.
Drives: group-average PC1 (as in Losorelli 2017) and each participant's own
projection on that PC1 filter (single-listen).  Whole song minus the first 10 s.
"""
import argparse, json, os, sys, time
import numpy as np
import h5py
from scipy.signal import resample_poly
from nmedt_pilot import load_song, pc1_component, NOMINAL_BPM


def harmonic_lock(r, phase, ks=(1, 2, 4)):
    rc = r - r.mean()
    return np.array([np.abs(np.mean(rc * np.exp(-1j * k * phase))) for k in ks])


def lock_z(r, phase, rng, n_null, fs, ks=(1, 2, 4)):
    obs = harmonic_lock(r, phase, ks)
    ms = int(10 * fs)
    null = np.array([harmonic_lock(np.roll(r, rng.randint(ms, len(r) - ms)), phase, ks) for _ in range(n_null)])
    return obs, (obs - null.mean(0)) / (null.std(0) + 1e-15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mat'); ap.add_argument('--repo', default='/home/user/dissoAdventureExperiments')
    ap.add_argument('--n-null', type=int, default=200); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--participants', type=int, default=20); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, 'chapter5Experiments'))
    from experiment_zero import LIFReservoir

    song = int(os.path.basename(a.mat).split('_')[0].replace('song', ''))
    f_beat = NOMINAL_BPM[song] / 60.0
    data, fs = load_song(a.mat)
    pc1, w = pc1_component(data)
    fs2 = 256.0; start = 10.0
    rng = np.random.RandomState(a.seed)
    res = LIFReservoir(seed=a.seed)
    out = {'song': song, 'f_beat': f_beat, 'harmonics': [1, 2, 4], 'drives': {}}

    def run_drive(name, sig):
        x = resample_poly(sig[int(start * fs):].astype(np.float64), 256, int(fs))
        x = (x - x.mean()) / (x.std() + 1e-12)
        t = np.arange(len(x)) / fs2 + start
        phase = 2 * np.pi * f_beat * t
        t0 = time.time(); spikes = res.forward(x); dt = time.time() - t0
        rate = spikes.mean(1)
        lx, zx = lock_z(x, phase, rng, a.n_null, fs2)
        lr, zr = lock_z(rate, phase, rng, a.n_null, fs2)
        print(f"[{name:>8}] T={len(x)/fs2:.0f}s rate={spikes.mean():.3f} ({dt:.1f}s) | drive z@(1,2,4)f = {np.round(zx,1)} | reservoir z = {np.round(zr,1)}")
        out['drives'][name] = {'rate': float(spikes.mean()), 'drive_z': zx.tolist(), 'res_z': zr.tolist(),
                               'drive_lock': lx.tolist(), 'res_lock': lr.tolist()}

    run_drive('groupPC1', pc1)
    for p in range(min(a.participants, data.shape[2])):
        xp = w @ (data[:, :, p] - data[:, :, p].mean(1, keepdims=True))
        run_drive(f'P{p+1:02d}', xp)

    zs_d = np.array([v['drive_z'] for k, v in out['drives'].items() if k.startswith('P')])
    zs_r = np.array([v['res_z'] for k, v in out['drives'].items() if k.startswith('P')])
    print("single-listen summary (20 participants): fraction with z>1.96 per harmonic")
    print("  drive    :", np.round((zs_d > 1.96).mean(0), 2), " median z", np.round(np.median(zs_d, 0), 2))
    print("  reservoir:", np.round((zs_r > 1.96).mean(0), 2), " median z", np.round(np.median(zs_r, 0), 2))
    out['summary'] = {'drive_frac_sig': (zs_d > 1.96).mean(0).tolist(), 'res_frac_sig': (zs_r > 1.96).mean(0).tolist(),
                      'drive_median_z': np.median(zs_d, 0).tolist(), 'res_median_z': np.median(zs_r, 0).tolist()}
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
