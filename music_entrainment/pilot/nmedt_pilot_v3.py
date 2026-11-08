#!/usr/bin/env python3
"""NMED-T pilot v3: windowed frequency-tagging SNR on drive vs reservoir rate.

1. Refine the beat frequency from the group PC1 whole-song spectrum: pick the
   strongest harmonic peak within +-0.6% of the nominal tempo (report which
   harmonic and the implied BPM).  Audio-free workaround; flagged as such.
2. Tagging statistic: Welch PSD (segment 40 s, Hann, 50% overlap); z of the bin
   at k*f_beat against the mean/sd of neighbouring bins 0.08-0.5 Hz away,
   excluding other harmonics.  Applied identically to the input drive x_t and the
   reservoir population rate r_t (and to the rate of a reservoir driven by a
   phase-randomised copy of x_t, as a nonlinearity-free surrogate).
3. Group-average PC1 drive and each participant's single-listen PC1 projection.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.signal import resample_poly, welch
from nmedt_pilot import load_song, pc1_component, NOMINAL_BPM


def refine_beat_freq(x, fs, f_nom, tol=0.006, harmonics=(1, 2, 4, 8)):
    n = int(2 ** np.ceil(np.log2(len(x) * 4)))
    X = np.abs(np.fft.rfft(x - x.mean(), n=n)); f = np.fft.rfftfreq(n, 1 / fs)
    best = None
    for k in harmonics:
        lo, hi = k * f_nom * (1 - tol), k * f_nom * (1 + tol)
        m = (f >= lo) & (f <= hi)
        if not m.any():
            continue
        i = np.argmax(X[m]); fp = f[m][i]
        nb = (f > k * f_nom * (1 - 0.05)) & (f < k * f_nom * (1 + 0.05)) & ~m
        z = (X[m][i] - X[nb].mean()) / (X[nb].std() + 1e-12)
        if best is None or z > best['z']:
            best = {'k': k, 'f_peak': float(fp), 'z': float(z), 'f_beat': float(fp / k)}
    return best


def tag_z(x, fs, f_beat, ks=(1, 2, 4, 8), seg_s=40):
    nper = int(fs * seg_s)
    f, p = welch(x - x.mean(), fs=fs, nperseg=nper, noverlap=nper // 2, window='hann')
    zs = []
    for k in ks:
        f0 = k * f_beat; i = np.argmin(np.abs(f - f0))
        nb = (np.abs(f - f0) > 0.08) & (np.abs(f - f0) < 0.5)
        for kk in ks:                       # keep other harmonics out of the baseline
            nb &= np.abs(f - kk * f_beat) > 0.08
        zs.append(float((p[i] - p[nb].mean()) / (p[nb].std() + 1e-15)))
    return np.array(zs)


def phase_randomize(sig, rng):
    F = np.fft.rfft(sig); ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0
    if len(sig) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(sig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mat'); ap.add_argument('--repo', default='/home/user/dissoAdventureExperiments')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--participants', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, 'chapter5Experiments'))
    from experiment_zero import LIFReservoir

    song = int(os.path.basename(a.mat).split('_')[0].replace('song', ''))
    f_nom = NOMINAL_BPM[song] / 60.0
    data, fs = load_song(a.mat)
    pc1, w = pc1_component(data)
    ref = refine_beat_freq(pc1[int(10 * fs):], fs, f_nom)
    f_beat = ref['f_beat']
    print(f"song {song}: nominal {f_nom:.4f} Hz ({NOMINAL_BPM[song]} BPM); refined from harmonic k={ref['k']} peak {ref['f_peak']:.4f} Hz (z={ref['z']:.1f}) -> {f_beat:.4f} Hz = {60*f_beat:.3f} BPM ({(f_beat/f_nom-1)*100:+.3f}%)")
    ks = (1, 2, 4, 8)
    fs2 = 256.0; rng = np.random.RandomState(a.seed); res = LIFReservoir(seed=a.seed)
    out = {'song': song, 'f_nominal': f_nom, 'refined': ref, 'harmonics': list(ks), 'drives': {}}

    def run(name, sig):
        x = resample_poly(sig[int(10 * fs):].astype(np.float64), 256, int(fs))
        x = (x - x.mean()) / (x.std() + 1e-12)
        zx = tag_z(x, fs2, f_beat, ks)
        rate = res.forward(x).mean(1); zr = tag_z(rate, fs2, f_beat, ks)
        xs = phase_randomize(x, rng); rs = res.forward(xs).mean(1); zs = tag_z(rs, fs2, f_beat, ks)
        print(f"[{name:>8}] drive z@(1,2,4,8)f={np.round(zx,1)} | reservoir rate z={np.round(zr,1)} | reservoir on phase-rand drive z={np.round(zs,1)}  rate={rate.mean():.3f}")
        out['drives'][name] = {'drive_z': zx.tolist(), 'res_z': zr.tolist(), 'res_surr_z': zs.tolist(), 'rate': float(rate.mean())}

    run('groupPC1', pc1)
    for p in range(min(a.participants, data.shape[2])):
        run(f'P{p+1:02d}', w @ (data[:, :, p] - data[:, :, p].mean(1, keepdims=True)))
    P = [k for k in out['drives'] if k.startswith('P')]
    zd = np.array([out['drives'][k]['drive_z'] for k in P]); zr = np.array([out['drives'][k]['res_z'] for k in P]); zs = np.array([out['drives'][k]['res_surr_z'] for k in P])
    print(f"single-listen (n={len(P)}): fraction z>1.96 per harmonic (1,2,4,8)f and median z")
    for nm, arr in (('drive', zd), ('reservoir', zr), ('res/phase-rand', zs)):
        print(f"  {nm:>15}: frac {np.round((arr>1.96).mean(0),2)}  median {np.round(np.median(arr,0),2)}")
    out['summary'] = {nm: {'frac_sig': (arr > 1.96).mean(0).tolist(), 'median_z': np.median(arr, 0).tolist()} for nm, arr in (('drive', zd), ('reservoir', zr), ('res_surr', zs))}
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
