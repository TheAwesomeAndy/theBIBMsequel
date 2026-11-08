#!/usr/bin/env python3
"""NMED-T week-1 feasibility pilot (audio-free).

What this checks, per song file:
  1. Loads the cleaned NMED-T file (125 ch x T x 20 participants, 125 Hz).
  2. Reproduces the Losorelli et al. (2017) sanity check: magnitude spectrum of
     the participant-averaged PC1 component shows a peak at the nominal beat
     frequency (and/or its harmonics).  If this fails, nothing downstream matters.
  3. Resamples a drive (PC1, and a fronto-central electrode cluster) to 256 Hz,
     z-scores it, and runs the canonical ARSPI-Net LIFReservoir (experiment_zero)
     on a continuous 60 s segment: reports firing rate and wall time.
  4. Beat-phase modulation index (Tort MI, 6 phase bins) of the population rate
     against the nominal beat grid, versus a circular-shift null.  MI is invariant
     to the unknown beat-phase origin, so it needs only the period and the song
     onset, not the audio.
Nothing here is a confirmatory result; it is a go/no-go check.
"""
import argparse, json, sys, time, os
import numpy as np
import h5py
from scipy.signal import resample_poly, welch
from scipy.io import loadmat

NOMINAL_BPM = {21: 55.97, 22: 69.44, 23: 74.26, 24: 82.42, 25: 91.46,
               26: 96.15, 27: 108.70, 28: 120.00, 29: 128.21, 30: 150.00}


def load_song(path):
    """Return data (125, T, P) float32 and fs."""
    try:
        with h5py.File(path, 'r') as f:
            key = [k for k in f.keys() if k.startswith('data')][0]
            d = f[key][()]          # h5py gives MATLAB arrays transposed
            fs = float(np.array(f['fs'][()]).ravel()[0])
        d = np.ascontiguousarray(d.T) if d.shape[0] != 125 else d
    except OSError:
        m = loadmat(path)
        key = [k for k in m if k.startswith('data')][0]
        d = m[key]; fs = float(np.ravel(m['fs'])[0])
    return d.astype(np.float32), fs


def pc1_component(data):
    """Participant-average, then PC1 spatial filter (as in the NMED-T paper)."""
    avg = np.nanmean(data, axis=2)                 # 125 x T
    avg = avg - avg.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(avg, full_matrices=False)
    w = u[:, 0]
    return w @ avg, w                              # (T,), (125,)


def beat_peak_report(x, fs, f_beat, label):
    nper = int(fs * 40)                            # 0.025 Hz resolution
    f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    def z_at(f0):
        i = np.argmin(np.abs(f - f0))
        nb = (np.abs(f - f0) > 0.08) & (np.abs(f - f0) < 0.5)   # neighbourhood
        return float((p[i] - p[nb].mean()) / (p[nb].std() + 1e-12)), float(f[i])
    out = {}
    for name, mult in (('beat', 1), ('half', 0.5), ('double', 2), ('quad', 4)):
        z, fa = z_at(f_beat * mult); out[name] = {'f': round(fa, 3), 'z_vs_neighbours': round(z, 2)}
    band = (f > 0.4) & (f < 6)
    top = f[band][np.argsort(p[band])[-5:]][::-1]
    out['top5_peaks_0.4-6Hz'] = [round(float(t), 3) for t in top]
    print(f"[{label}] beat {f_beat:.3f} Hz: " + ", ".join(f"{k}={v['z_vs_neighbours']}z@{v['f']}" for k, v in out.items() if k != 'top5_peaks_0.4-6Hz') + f" | top peaks {out['top5_peaks_0.4-6Hz']}")
    return out


def tort_mi(phase, rate, n_bins=6):
    bins = np.floor((phase % (2 * np.pi)) / (2 * np.pi) * n_bins).astype(int)
    m = np.array([rate[bins == k].mean() if np.any(bins == k) else 0.0 for k in range(n_bins)])
    if m.sum() <= 0:
        return 0.0, m
    p = m / m.sum(); p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float((np.log(n_bins) - H) / np.log(n_bins)), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mat'); ap.add_argument('--repo', default='/home/user/dissoAdventureExperiments')
    ap.add_argument('--seconds', type=float, default=60); ap.add_argument('--start', type=float, default=15)
    ap.add_argument('--n-null', type=int, default=200); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, 'chapter5Experiments'))
    from experiment_zero import LIFReservoir

    song = int(os.path.basename(a.mat).split('_')[0].replace('song', ''))
    f_beat = NOMINAL_BPM[song] / 60.0
    t0 = time.time(); data, fs = load_song(a.mat)
    print(f"loaded {a.mat}: shape {data.shape}, fs {fs}, {time.time()-t0:.1f}s, nan frac {np.isnan(data).mean():.4f}")
    res = {'song': song, 'f_beat': f_beat, 'shape': list(data.shape), 'fs': fs}

    # 1) SS-EP sanity check on PC1 (whole song, participant average)
    pc1, w = pc1_component(data)
    res['ssep_pc1'] = beat_peak_report(pc1, fs, f_beat, 'PC1 avg')
    # and on the grand mean of all electrodes (ME filter in the paper)
    me = np.nanmean(np.nanmean(data, axis=2), axis=0)
    res['ssep_me'] = beat_peak_report(me, fs, f_beat, 'mean-electrode avg')

    # 2) Reservoir pilot on a 60 s segment of PC1 at 256 Hz
    seg = pc1[int(a.start * fs): int((a.start + a.seconds) * fs)]
    x = resample_poly(seg.astype(np.float64), 256, int(fs))
    x = (x - x.mean()) / (x.std() + 1e-12)
    fs2 = 256.0
    r = LIFReservoir(seed=a.seed)
    t0 = time.time(); spikes = r.forward(x); dt = time.time() - t0
    rate = spikes.mean(axis=1)                      # population rate per step
    print(f"reservoir: {len(x)} steps in {dt:.2f}s, mean firing rate {spikes.mean():.4f}, silent units {int((spikes.sum(0)==0).sum())}/256")
    res['reservoir'] = {'steps': len(x), 'seconds_wall': round(dt, 2), 'mean_rate': float(spikes.mean()),
                        'silent_units': int((spikes.sum(0) == 0).sum())}

    # 3) Beat-phase MI vs circular-shift null (phase origin arbitrary; MI is origin-invariant)
    t = np.arange(len(x)) / fs2 + a.start
    phase = 2 * np.pi * f_beat * t
    mi, hist = tort_mi(phase, rate)
    rng = np.random.RandomState(a.seed)
    min_shift = int(10 * fs2)
    null = np.array([tort_mi(phase, np.roll(rate, rng.randint(min_shift, len(rate) - min_shift)))[0] for _ in range(a.n_null)])
    z = (mi - null.mean()) / (null.std() + 1e-12)
    p = float((np.sum(null >= mi) + 1) / (len(null) + 1))
    print(f"beat-phase MI (population rate): {mi:.5f}; null mean {null.mean():.5f} sd {null.std():.5f}; z={z:.2f}, p={p:.3f}; phase-bin rates {np.round(hist,4)}")
    res['beat_phase_mi'] = {'mi': mi, 'null_mean': float(null.mean()), 'null_sd': float(null.std()), 'z': float(z), 'p': p}

    # same test on the linear drive itself (is the reservoir adding anything beyond the input's own beat-locking?)
    mi_in, _ = tort_mi(phase, x - x.min())
    null_in = np.array([tort_mi(phase, np.roll(x - x.min(), rng.randint(min_shift, len(x) - min_shift)))[0] for _ in range(a.n_null)])
    z_in = (mi_in - null_in.mean()) / (null_in.std() + 1e-12)
    print(f"beat-phase MI (input drive itself): {mi_in:.5f}; z={z_in:.2f}")
    res['beat_phase_mi_input'] = {'mi': mi_in, 'z': float(z_in)}

    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
