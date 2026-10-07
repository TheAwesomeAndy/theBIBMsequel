#!/usr/bin/env python3
"""Restricted-data gate: run before every commit (CLAUDE.md, Data isolation).

Checks the files staged for commit (or every tracked file with --all):
  * no data or derivative formats (.pkl .pickle .mat .npy .npz .dat .vhdr .vmrk .bdf .edf
    .fif .set .fdt .h5 .hdf5 .csv outside outputs/aggregate);
  * no file larger than 5 MB except PDFs under manuscript/ and the ARSPI/defense assets;
  * JSON under outputs/ holds no list whose length equals a per-observation or per-subject
    count of either cohort (633, 211, 684, 228, 32x... ) which would indicate subject-level
    values, and no subject identifiers (SHAPE ids, TCRZEM S### ids);
  * no Google Drive ids or private links.
Exit status 1 on any finding.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_EXT = {".pkl", ".pickle", ".mat", ".npy", ".npz", ".dat", ".vhdr", ".vmrk", ".bdf", ".edf",
           ".fif", ".set", ".fdt", ".h5", ".hdf5"}
SUSPECT_LEN = {633, 211, 684, 228}
ID_PAT = re.compile(r"\bS\d{3}\b|SHAPE_Community_\d+|drive\.google\.com|docs\.google\.com/.+/d/")


def staged(all_files: bool):
    cmd = ["git", "ls-files"] if all_files else ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"]
    return [p for p in subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.split("\n") if p]


def walk(obj, path, out):
    if isinstance(obj, list):
        if len(obj) in SUSPECT_LEN and all(isinstance(v, (int, float)) for v in obj):
            out.append(f"{path}: numeric list of length {len(obj)}")
        for i, v in enumerate(obj[:50]):
            walk(v, f"{path}[{i}]", out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}", out)


def main():
    files = staged("--all" in sys.argv)
    problems = []
    for f in files:
        p = ROOT / f
        if not p.exists():
            continue
        ext = p.suffix.lower()
        if ext in BAD_EXT:
            problems.append(f"{f}: data format {ext}")
        if ext == ".csv" and not f.startswith("outputs/aggregate/"):
            problems.append(f"{f}: CSV outside outputs/aggregate")
        size = p.stat().st_size
        if size > 5e6 and not (ext == ".pdf" or f.startswith(("defense_figures/", "ArspiNet"))):
            problems.append(f"{f}: {size / 1e6:.1f} MB")
        if ext in {".json", ".csv", ".md", ".tex", ".py", ".txt"}:
            text = p.read_text(errors="ignore")
            if f.endswith("check_restricted_data.py"):
                continue
            for m in ID_PAT.finditer(text):
                if ext == ".py" and m.group(0).startswith("S") and "S<id>" in text:
                    continue
                problems.append(f"{f}: identifier or private link '{m.group(0)}'")
                break
        if ext == ".json" and f.startswith("outputs/"):
            out = []
            walk(json.load(open(p)), f, out)
            problems += out
    for pr in problems:
        print("[restricted] " + pr)
    print(f"[restricted] checked {len(files)} file(s): {'FAIL' if problems else 'PASS'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
