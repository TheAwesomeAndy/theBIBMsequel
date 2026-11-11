"""Shared experiment plumbing: data loading, encoder fold functions, provenance, output.

Every experiment writes ONE aggregate JSON to outputs/aggregate/. Nothing subject-level is
ever written. Restricted inputs come from --shape-dir / $SHAPE_DIR or --pickle / $SHAPE_PICKLE.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sequel import config as C  # noqa: E402
from sequel import data, features, fills, protocol, theory  # noqa: E402

OUT = ROOT / "outputs" / "aggregate"


# ------------------------------------------------------------------------------ CLI / IO
def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--shape-dir", help="three-condition SHAPE directory (or $SHAPE_DIR)")
    p.add_argument("--pickle", help="BIBM feature pickle (or $SHAPE_PICKLE); legacy signal only")
    p.add_argument("--out", help="output JSON name (default set per experiment)")
    p.add_argument("--quick", action="store_true",
                   help="one CV seed and 500 bootstrap resamples, for smoke runs only")
    p.add_argument("--synthetic", action="store_true",
                   help="run on the synthetic test cohort (code check; numbers are meaningless)")
    return p


def load_raw(args):
    """Raw SHAPE epochs (or the synthetic cohort). Returns raw, y, groups."""
    if args.synthetic:
        sys.path.insert(0, str(ROOT))
        from tests.synthetic import raw_epochs
        return raw_epochs(n_subj=24)
    return data.load_shape_raw(args.shape_dir)


def provenance(args) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:
            return None
    pk = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "argv": sys.argv, "python": platform.python_version(), "platform": platform.platform(),
            "packages": pk, "synthetic": bool(getattr(args, "synthetic", False)),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write(result: dict, name: str, args) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if getattr(args, "synthetic", False):
        name = name.replace(".json", "_SYNTHETIC.json")
    result = {"provenance": provenance(args), **result}
    path = OUT / name
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"[out] wrote {path}")
    return path


def cv_settings(args):
    seeds = (C.CV_SEEDS[0],) if args.quick else C.CV_SEEDS
    n_boot = 500 if args.quick else C.N_BOOT
    return seeds, n_boot


# ------------------------------------------------------------------ encoder fold function
def removal_draw(n_ch: int, frac: float, seed: int, fold: int) -> np.ndarray:
    """Identical to BIBM analysis/perturbations.py (same draws for every encoder)."""
    if frac <= 0:
        return np.array([], int)
    rng = np.random.default_rng(1000 * seed + fold)
    return rng.choice(n_ch, int(round(frac * n_ch)), replace=False)


def encoder_fold_fn(blocks: np.ndarray, use_pca: bool, drop_frac: float = 0.3,
                    rules=("zero", "mean", "knn", "spatial"), positions=None, refit: bool = True):
    """Fold function producing clean, feature-fill, signal-removal and refit conditions.

    ``blocks`` (n_obs, n_ch, f) are label-free encoder outputs. With ``use_pca`` a BlockPCA
    is fitted on the training observations of each fold (BIBM train-only PCA). A removed
    electrode is a zero trace; every encoder here maps a zero trace to a zero block, so
    signal-level removal equals placing the embedding of a zero block (``sig``).
    """
    pct = int(round(drop_frac * 100))
    n_ch = blocks.shape[1]

    def fn(seed, fold, tr, te):
        if use_pca:
            P = features.BlockPCA().fit(blocks[tr])
            E, v0 = P.transform(blocks), P.silent_channel()
        else:
            E, v0 = blocks.astype(np.float64), np.zeros(blocks.shape[2])
        f = E.shape[2]
        Xtr = E[tr].reshape(len(tr), -1)
        conds = {"clean": (Xtr, E[te].reshape(len(te), -1), None)}
        if drop_frac > 0:
            drop = removal_draw(n_ch, drop_frac, seed, fold)
            meta = {"drop": drop, "n_feat": f, "Xtr": Xtr}
            for rule in rules:
                if rule == "spline" and (positions is None or use_pca):
                    continue
                Ete = fills.fill_blocks(E[te], E[tr], drop, rule, positions)
                conds[f"fc{pct}_{rule}"] = (Xtr, Ete.reshape(len(te), -1), meta)
            Ete = E[te].copy()
            Ete[:, drop, :] = v0
            conds[f"sig{pct}"] = (Xtr, Ete.reshape(len(te), -1), None)
            if refit:
                keep = np.setdiff1d(np.arange(n_ch), drop)
                conds[f"refit{pct}"] = (E[tr][:, keep].reshape(len(tr), -1),
                                        E[te][:, keep].reshape(len(te), -1), None)
        return conds

    return fn


class TheoryLog:
    """Hook collecting Proposition-1 quantities for every zero-fill condition."""

    def __init__(self):
        self.rows: dict[str, list] = {}

    def __call__(self, name, seed, fold, sc, clf, meta):
        if meta is None or not name.endswith("_zero"):
            return
        cols = theory.block_cols(meta["drop"], meta["n_feat"])
        self.rows.setdefault(name, []).append({
            "non_centeredness": theory.non_centeredness(sc),
            "dropped_non_centeredness": float(np.mean(np.abs(sc.mean_[cols]) / sc.scale_[cols])),
            "logit_shift_spread": theory.centered_spread(theory.zero_fill_shift(sc, clf, cols)),
            "fill_bias_index": theory.fill_bias_index(sc, clf, cols, meta["Xtr"]),
        })

    def summary(self) -> dict:
        return {k: {q: float(np.mean([r[q] for r in v])) for q in v[0]} for k, v in self.rows.items()}
