#!/usr/bin/env python3
"""Phase 2. Download public datasets and build condition averages in the common format.

    SEQUEL_DATA=/path/data_local python experiments/fetch_datasets.py erpcore_P3 erpcore_N400 erpcore_N170 tcrzem

Raw downloads and the cached averages stay under $SEQUEL_DATA. The only repository output
is outputs/aggregate/datasets_summary.json (counts, channels, trial numbers: no signals).
With --delete-raw the large ERP CORE .fdt/.set files are removed after averaging.
"""
from __future__ import annotations

import json
import sys
import time

from common import OUT, provenance  # noqa: E402
from sequel import datasets  # noqa: E402


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+", choices=[k for k in datasets.REGISTRY if k != "shape"])
    p.add_argument("--subjects", type=int, nargs="+", help="ERP CORE subject numbers (default all)")
    p.add_argument("--delete-raw", action="store_true")
    args = p.parse_args()
    summary_path = OUT / "datasets_summary.json"
    for name in args.names:
        summary = {}
        t0 = time.time()
        if name.startswith("erpcore_"):
            task = name.split("_", 1)[1]
            datasets.fetch_erpcore(task, args.subjects)
            s = datasets.build_erpcore(task, delete_raw=args.delete_raw)
        else:
            datasets.fetch_tcrzem()
            s = datasets.build_tcrzem()
        s.save(datasets.DATA_ROOT / "cache" / f"{name}.npz")
        summary[name] = s.summary() | {"skipped_subjects": s.info.get("skipped_subjects"),
                                       "settings": {k: v for k, v in s.info.items() if k != "skipped_subjects"},
                                       "build_seconds": round(time.time() - t0)}
        print(f"[{name}] {summary[name]}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        # re-read just before writing so concurrent runs for other datasets are not lost
        prev = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        prev.pop("provenance", None)
        prev[name] = summary[name]
        summary_path.write_text(json.dumps({"provenance": provenance(args)} | prev, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
