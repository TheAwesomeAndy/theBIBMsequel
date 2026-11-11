"""Condition-averaged ERP datasets in one common format.

Every dataset becomes an ``ERPSet``: one trial-averaged ERP per subject and condition
(the SHAPE design), stored at the native rate with the stimulus onset index, plus channel
names and, where known, electrode positions. ``ERPSet.to_epochs`` sends it through the same
``data.preprocess`` path as SHAPE (FIR decimation to 256 Hz, then normalization), so every
encoder, fill rule and validation step applies unchanged.

Datasets
    shape     SHAPE three-condition (restricted; montage not yet available)
    erpcore   ERP CORE (Kappenman et al., 2021; OSF 9f5w7, BIDS raw files, CC-BY):
              P3 (rare vs frequent), N400 (related vs unrelated), N170 (faces vs cars)
    tcrzem    IAPS passive viewing, DataverseNL doi:10.34894/TCRZEM: Neg/Neu/Pos

Downloads and the averaged arrays live under a local data root ($SEQUEL_DATA, default
~/data_local), never in the repository.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import data

DATA_ROOT = Path(os.environ.get("SEQUEL_DATA", Path.home() / "data_local"))

# Epoch layout shared with SHAPE: -205 .. +1023 samples at 1024 Hz (-200 .. +999 ms).
PRE_MS, POST_MS = 200.0, 1000.0


@dataclass
class ERPSet:
    name: str
    X: np.ndarray                 # (n_obs, n_times, n_ch) at fs_raw, microvolts
    y: np.ndarray
    groups: np.ndarray
    fs_raw: int
    onset_raw: int
    ch_names: list[str]
    cond_names: list[str]
    positions: np.ndarray | None = None      # (n_ch, 3), any scale (projected to a sphere)
    n_trials: np.ndarray | None = None       # trials averaged into each observation
    info: dict = field(default_factory=dict)

    def to_epochs(self, decimation: str = "fir", norm: str = "baseline_z") -> data.Epochs:
        ep = data.preprocess(self.X, self.y, self.groups, decimation, norm,
                             fs_raw=self.fs_raw, onset_raw=self.onset_raw, source=self.name)
        ep.info.update(ch_names=self.ch_names, cond_names=self.cond_names)
        return ep

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, y=self.y, groups=self.groups,
                            positions=self.positions if self.positions is not None else np.zeros(0),
                            n_trials=self.n_trials if self.n_trials is not None else np.zeros(0),
                            meta=json.dumps({"name": self.name, "fs_raw": self.fs_raw,
                                             "onset_raw": self.onset_raw, "ch_names": self.ch_names,
                                             "cond_names": self.cond_names, "info": self.info}))

    @classmethod
    def load(cls, path: Path) -> "ERPSet":
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        pos = z["positions"] if z["positions"].size else None
        nt = z["n_trials"] if z["n_trials"].size else None
        return cls(m["name"], z["X"], z["y"], z["groups"], m["fs_raw"], m["onset_raw"],
                   m["ch_names"], m["cond_names"], pos, nt, m["info"])

    def summary(self) -> dict:
        return {"name": self.name, "n_obs": int(len(self.y)), "n_subjects": int(len(set(self.groups))),
                "n_channels": len(self.ch_names), "conditions": self.cond_names,
                "fs_raw": self.fs_raw, "has_positions": self.positions is not None,
                "trials_per_average_median": float(np.median(self.n_trials)) if self.n_trials is not None else None}


# --------------------------------------------------------------------------- shared tools
def _curl(url: str, dest: Path | None = None, timeout: int = 600) -> bytes | None:
    cmd = ["curl", "-sgfL", "--retry", "4", "--retry-delay", "2", "-m", str(timeout), url]
    if dest is None:
        return subprocess.check_output(cmd)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.check_call(cmd + ["-o", str(tmp)])
    tmp.rename(dest)
    return None


def cut_epochs(cont: np.ndarray, sfreq: float, onsets_s: np.ndarray,
               pre_ms: float = PRE_MS, post_ms: float = POST_MS):
    """Cut (n_ev, T, ch) epochs from continuous (n_times, ch) data; baseline-correct each.

    Lengths are ``round(ms * sfreq / 1000)``; at 1024 Hz that is 205 + 1024 = 1229 samples
    with onset at sample 205, the SHAPE layout. Events whose epoch leaves the recording are
    dropped; the returned mask says which events were kept.
    """
    n_pre = int(round(pre_ms * sfreq / 1000.0))
    n_post = int(round(post_ms * sfreq / 1000.0))
    T = n_pre + n_post
    starts = np.round(onsets_s * sfreq).astype(int) - n_pre
    ok = (starts >= 0) & (starts + T <= cont.shape[0])
    ep = np.stack([cont[s:s + T] for s in starts[ok]]) if ok.any() else np.zeros((0, T, cont.shape[1]))
    ep = ep - ep[:, :n_pre].mean(axis=1, keepdims=True)
    return ep, n_pre, ok


def average_conditions(trials: np.ndarray, conds: np.ndarray, cond_ids: list[int],
                       reject_uv: float | None = 200.0, min_trials: int = 10):
    """Per-condition averages after peak-to-peak rejection. None if any condition is short."""
    if reject_uv is not None and len(trials):
        keep = (trials.max(axis=1) - trials.min(axis=1)).max(axis=1) < reject_uv
        trials, conds = trials[keep], conds[keep]
    out, counts = [], []
    for c in cond_ids:
        m = conds == c
        if m.sum() < min_trials:
            return None, None
        out.append(trials[m].mean(axis=0))
        counts.append(int(m.sum()))
    return np.stack(out), counts


# -------------------------------------------------------------------------------- SHAPE
def shape(shape_dir: str | None = None) -> ERPSet:
    raw, y, g = data.load_shape_raw(shape_dir)
    return ERPSet("shape", raw, y, g, C.FS_RAW, C.ONSET_RAW, [f"ch{i:02d}" for i in range(raw.shape[2])],
                  ["Neg", "Neu", "Pos"], None, None, {"montage": "unknown (requested from the lab)"})


# ----------------------------------------------------------------------------- ERP CORE
ERPCORE_NODE = "9f5w7"
ERPCORE_TASKS = {
    # task: (conditions as {label: predicate on event value}, reference channels or 'average')
    "P3": ({"frequent": lambda v: 11 <= v <= 55 and v // 10 != v % 10,
            "rare": lambda v: v in (11, 22, 33, 44, 55)}, ["P9", "P10"]),
    "N400": ({"related": lambda v: v in (211, 212), "unrelated": lambda v: v in (221, 222)},
             ["P9", "P10"]),
    "N170": ({"faces": lambda v: 1 <= v <= 40, "cars": lambda v: 41 <= v <= 80}, "average"),
}
_ERPCORE_SUFFIXES = ("_eeg.set", "_eeg.fdt", "_events.tsv", "_electrodes.tsv", "_channels.tsv")


def _osf_list(href: str) -> list[dict]:
    out, url = [], href + ("&" if "?" in href else "?") + "page[size]=100"
    while url:
        j = json.loads(_curl(url, timeout=120))
        out += j["data"]
        url = j["links"].get("next")
    return out


def _download_all(jobs: list[tuple[str, Path, int]], workers: int = 8) -> None:
    """Fetch (url, dest, expected_size) jobs in parallel, skipping complete files."""
    from concurrent.futures import ThreadPoolExecutor

    import time

    def one(job):
        url, dest, _ = job
        for attempt in range(6):                # backoff on HTTP 429/5xx from the hosts
            try:
                return _curl(url, dest)
            except subprocess.CalledProcessError:
                time.sleep(5 * 2 ** attempt)
        raise RuntimeError(f"download failed after retries: {url} -> {dest}")

    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size == j[2])]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def fetch_erpcore(task: str, subjects: list[int] | None = None, root: Path = DATA_ROOT,
                  workers: int = 4) -> Path:
    """Download the BIDS raw files of one ERP CORE task. Returns the local task directory."""
    if task not in ERPCORE_TASKS:
        raise ValueError(f"task must be one of {list(ERPCORE_TASKS)}")
    base = root / "erpcore"
    top = _osf_list(f"https://api.osf.io/v2/nodes/{ERPCORE_NODE}/files/osfstorage/")
    bids = _osf_list(top[0]["relationships"]["files"]["links"]["related"]["href"])
    subs = sorted((n for n in bids if n["attributes"]["name"].startswith("sub-")),
                  key=lambda n: n["attributes"]["name"])
    if subjects is not None:
        want = {f"sub-{s:03d}" for s in subjects}
        subs = [n for n in subs if n["attributes"]["name"] in want]
    jobs = []
    for sub in subs:
        sname = sub["attributes"]["name"]
        sessions = _osf_list(sub["relationships"]["files"]["links"]["related"]["href"])
        ses = [n for n in sessions if n["attributes"]["name"] == f"ses-{task}"]
        if not ses:
            continue
        eeg = _osf_list(ses[0]["relationships"]["files"]["links"]["related"]["href"])[0]
        for f in _osf_list(eeg["relationships"]["files"]["links"]["related"]["href"]):
            nm = f["attributes"]["name"]
            dest = base / task / sname / nm
            if nm.endswith(_ERPCORE_SUFFIXES):
                jobs.append((f["links"]["download"], dest, f["attributes"]["size"]))
    _download_all(jobs, workers)
    return base / task


def build_erpcore(task: str, root: Path = DATA_ROOT, highpass: float = 0.1,
                  reject_uv: float = 200.0, min_trials: int = 10, delete_raw: bool = False) -> ERPSet:
    """Condition averages for one ERP CORE task from the downloaded BIDS files."""
    import mne
    import pandas as pd

    conds, ref = ERPCORE_TASKS[task]
    labels = list(conds)
    tdir = root / "erpcore" / task
    X, y, g, ntr, ch_names, pos, skipped = [], [], [], [], None, None, []
    for sdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
        stem = f"{sdir.name}_ses-{task}_task-{task}"
        raw = mne.io.read_raw_eeglab(sdir / f"{stem}_eeg.set", preload=True, verbose="error")
        raw.pick([c for c in raw.ch_names if "EOG" not in c.upper()])
        if ref == "average":
            raw.set_eeg_reference("average", verbose="error")
        else:
            raw.set_eeg_reference(ref, verbose="error")
        raw.filter(highpass, None, verbose="error")
        cont = raw.get_data().T * 1e6                                   # (n_times, ch) in uV
        ev = pd.read_csv(sdir / f"{stem}_events.tsv", sep="\t")
        ev = ev[ev["trial_type"] == "stimulus"]
        vals = ev["value"].astype(int).to_numpy()
        lab = np.full(len(vals), -1)
        for k, (name, pred) in enumerate(conds.items()):
            lab[[pred(v) for v in vals]] = k
        sel = lab >= 0
        trials, n_pre, kept = cut_epochs(cont, raw.info["sfreq"], ev["onset"].to_numpy()[sel])
        lab = lab[sel][kept]
        avg, counts = average_conditions(trials, lab, list(range(len(labels))), reject_uv, min_trials)
        if avg is None:
            skipped.append(sdir.name)
            continue
        if ch_names is None:
            ch_names = raw.ch_names
            el = pd.read_csv(sdir / f"{stem}_electrodes.tsv", sep="\t").set_index("name")
            pos = el.loc[ch_names, ["x", "y", "z"]].to_numpy(float)
        elif raw.ch_names != ch_names:
            raise ValueError(f"{sdir.name}: channel set differs")
        X += list(avg)
        y += list(range(len(labels)))
        g += [sdir.name] * len(labels)
        ntr += counts
        if delete_raw:
            for p in sdir.glob("*_eeg.*"):
                p.unlink()
    return ERPSet(f"erpcore_{task}", np.asarray(X), np.asarray(y), np.asarray(g),
                  int(raw.info["sfreq"]), n_pre, ch_names, labels, pos, np.asarray(ntr),
                  {"reference": ref, "highpass_hz": highpass, "reject_uv": reject_uv,
                   "min_trials": min_trials, "skipped_subjects": skipped,
                   "source": "OSF 9f5w7 (ERP CORE BIDS raw files)"})


# ------------------------------------------------------------------------------- TCRZEM
TCRZEM_DOI = "doi:10.34894/TCRZEM"
_TCRZEM_RE = re.compile(r"(S\d+)_Mastoid_(Neg|Neu|Pos)_(High|Low)\.(dat|vhdr|vmrk)$", re.I)
TCRZEM_VALENCE = {"Neg": 0, "Neu": 1, "Pos": 2}


def fetch_tcrzem(root: Path = DATA_ROOT, max_subjects: int | None = None, workers: int = 8) -> Path:
    """Download the exported, mastoid-referenced segment files (about 4.3 GB)."""
    idx = json.loads(_curl(f"https://dataverse.nl/api/datasets/:persistentId/?persistentId={TCRZEM_DOI}",
                           timeout=120))["data"]["latestVersion"]["files"]
    files = [f for f in idx if _TCRZEM_RE.search(f["dataFile"]["filename"])]
    if max_subjects is not None:
        keep = sorted({_TCRZEM_RE.search(f["dataFile"]["filename"]).group(1) for f in files})[:max_subjects]
        files = [f for f in files if _TCRZEM_RE.search(f["dataFile"]["filename"]).group(1) in keep]
    out = root / "tcrzem"
    _download_all([(f"https://dataverse.nl/api/access/datafile/{f['dataFile']['id']}",
                    out / f["dataFile"]["filename"], f["dataFile"]["filesize"]) for f in files], workers)
    return out


def _vhdr(path: Path) -> dict:
    """Minimal BrainVision header reader: channels, resolutions, rate, segment length."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    sec = None
    names, res = [], []
    hdr = {}
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("["):
            sec = line
            continue
        if "=" not in line or line.startswith(";"):
            continue
        k, v = line.split("=", 1)
        if sec == "[Channel Infos]" and k.startswith("Ch"):
            parts = v.split(",")
            names.append(parts[0])
            res.append(float(parts[2]) if len(parts) > 2 and parts[2] else 1.0)
        else:
            hdr[k.strip().lower()] = v.strip()
    return {"names": names, "res": np.asarray(res), "sfreq": 1e6 / float(hdr["samplinginterval"]),
            "segment_points": int(hdr.get("segmentdatapoints", 0)) or None,
            "binary": hdr.get("binaryformat", "INT_16"), "orientation": hdr.get("dataorientation")}


def build_tcrzem(root: Path = DATA_ROOT, pre_stim_ms: float = 1000.0, reject_uv: float = 200.0,
                 min_trials: int = 4) -> ERPSet:
    """Valence averages (Neg/Neu/Pos, arousal pooled) per subject from the exported segments.

    Segments are 1792 samples at 512 Hz spanning -1000..+2500 ms (dissertation loader); they
    are re-cut to -200..+1000 ms. Non-EEG channels are dropped by name. Positions come from
    the standard 10-05 template for the channel names (MNE), not from digitized locations.
    """
    tdir = root / "tcrzem"
    by_subj: dict[str, dict[int, list[Path]]] = {}
    for p in sorted(tdir.glob("*.vhdr")):
        m = _TCRZEM_RE.search(p.name)
        if m:
            by_subj.setdefault(m.group(1), {}).setdefault(TCRZEM_VALENCE[m.group(2)], []).append(p)
    X, y, g, ntr, skipped, ch_names = [], [], [], [], [], None
    for sid, conds in sorted(by_subj.items()):
        trials_by = {}
        for c, hdrs in conds.items():
            segs = []
            for h in hdrs:
                info = _vhdr(h)
                dt = {"INT_16": "<i2", "IEEE_FLOAT_32": "<f4"}[info["binary"]]
                raw = np.fromfile(h.with_suffix(".dat"), dtype=dt).astype(np.float64)
                n_ch = len(info["names"])
                seg_pts = info["segment_points"] or 1792
                d = raw[: (raw.size // (n_ch * seg_pts)) * n_ch * seg_pts].reshape(-1, seg_pts, n_ch)
                segs.append(d * info["res"][None, None, :])
            eeg = [i for i, n in enumerate(info["names"]) if not re.search(r"EOG|ECG|EMG|REF|M[12]$|^A[12]$", n, re.I)]
            names = [info["names"][i] for i in eeg]
            sf = info["sfreq"]
            onset = int(round(pre_stim_ms * sf / 1000.0))
            n_pre, n_post = int(round(PRE_MS * sf / 1000.0)), int(round(POST_MS * sf / 1000.0))
            tr = np.concatenate(segs)[:, onset - n_pre:onset + n_post, :][:, :, eeg]
            trials_by[c] = tr - tr[:, :n_pre].mean(axis=1, keepdims=True)
        trials = np.concatenate([trials_by[c] for c in sorted(trials_by)])
        labs = np.concatenate([[c] * len(trials_by[c]) for c in sorted(trials_by)])
        avg, counts = average_conditions(trials, labs, [0, 1, 2], reject_uv, min_trials)
        if avg is None:
            skipped.append(sid)
            continue
        if ch_names is None:
            ch_names = names
        elif names != ch_names:
            skipped.append(sid)
            continue
        X += list(avg)
        y += [0, 1, 2]
        g += [sid] * 3
        ntr += counts
    return ERPSet("tcrzem", np.asarray(X), np.asarray(y), np.asarray(g), int(sf), n_pre, ch_names,
                  ["Neg", "Neu", "Pos"], template_positions(ch_names), np.asarray(ntr),
                  {"reject_uv": reject_uv, "min_trials": min_trials, "skipped_subjects": skipped,
                   "source": TCRZEM_DOI, "positions": "standard_1005 template"})


def template_positions(ch_names: list[str]) -> np.ndarray | None:
    """Template 3-D positions for 10-20/10-05 names (MNE). None if any name is unknown."""
    try:
        import mne
    except ImportError:
        return None
    mont = mne.channels.make_standard_montage("standard_1005")
    lut = {k.lower(): v for k, v in mont.get_positions()["ch_pos"].items()}
    try:
        return np.asarray([lut[n.lower()] for n in ch_names])
    except KeyError:
        return None


REGISTRY = {"shape": shape, "erpcore_P3": lambda **k: build_erpcore("P3", **k),
            "erpcore_N400": lambda **k: build_erpcore("N400", **k),
            "erpcore_N170": lambda **k: build_erpcore("N170", **k), "tcrzem": build_tcrzem}


def cached(name: str, root: Path = DATA_ROOT, **kw) -> ERPSet:
    """Build once and cache the averages as root/cache/<name>.npz (local, never committed)."""
    path = root / "cache" / f"{name}.npz"
    if path.exists():
        return ERPSet.load(path)
    s = REGISTRY[name](**kw)
    s.save(path)
    return s
