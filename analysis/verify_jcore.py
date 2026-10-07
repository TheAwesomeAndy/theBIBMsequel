#!/usr/bin/env python3
"""Check that jcore.py reproduces the conference implementation on synthetic data.

Compares reservoir codes (rho = 0.9, 0, 1.2, chunked), band power, ERP windows, the
damage-spreading order parameter, and the subject bootstrap (point, CI, paired
difference) between jcore.py and the conference modules. No restricted data needed.
Exit status 1 on any mismatch.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jcore as J
import experiment_camera_ready as EC
import experiment2_rawsignal as E2S
import reanalysis_subject_bootstrap as RB
rng = np.random.default_rng(0)
X = rng.standard_normal((30, 256, 5))
t=time.time(); B1,r1 = EC.bsc6(X, 0.9); B2,r2 = J.reservoir_codes(X, 0.9)
print('bsc equal:', np.array_equal(B1, B2), r1, r2, time.time()-t)
B1,_ = EC.bsc6(X, 0.0); B2,_ = J.reservoir_codes(X, 0.0); print('bsc rho0 equal:', np.array_equal(B1,B2))
B2c,_ = J.reservoir_codes(X, 1.2, chunk=7); B1,_=EC.bsc6(X,1.2); print('chunked equal:', np.array_equal(B1,B2c))
bp1 = E2S.bandpower_all(X); bp2 = J.bandpower(X); print('bandpower max rel diff:', np.max(np.abs(bp1-bp2)/np.abs(bp1).max()))
print('erp equal:', np.allclose(E2S.erp_all(X), J.erp_windows(X)))
print('damage:', EC.damage(X, 0.9, n_drive=100), J.damage(X, 0.9, n_drive=100))
# bootstrap
y = np.tile([0,1,2], 50); g = np.repeat(np.arange(50), 3)
pa = rng.integers(0,3,150); pb = rng.integers(0,3,150)
p1, c1, _ = RB.subject_bootstrap(y, g, {'a':pa,'b':pb}, n_boot=300)
p2, c2, _ = J.boot_ci(y, g, {'a':pa,'b':pb}, n_boot=300)
print('boot:', p1, c1, '\n      ', p2, c2)
print('paired:', RB.paired_diff_ci(y, g, pa, pb, n_boot=300), J.paired_ci(y, g, pa, pb, n_boot=300))

ok = (np.array_equal(*[EC.bsc6(X, r)[0] for r in (0.9,)] + [J.reservoir_codes(X, 0.9)[0]])
      and np.array_equal(EC.bsc6(X, 0.0)[0], J.reservoir_codes(X, 0.0)[0])
      and np.array_equal(EC.bsc6(X, 1.2)[0], B2c)
      and np.max(np.abs(bp1 - bp2) / np.abs(bp1).max()) < 1e-12
      and np.allclose(E2S.erp_all(X), J.erp_windows(X))
      and EC.damage(X, 0.9, n_drive=100) == J.damage(X, 0.9, n_drive=100)
      and p1 == p2 and all(tuple(c1[k]) == tuple(c2[k]) for k in c1))
print("[verify_jcore]", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
