"""List decimal numbers in a manuscript that do not occur in any committed output.

Every float in outputs/**/*.json is rounded to 2, 3 and 4 decimals, together with
its absolute value and 1 minus the value. A manuscript number with three or more
decimals that matches none of them is printed for manual tracing. Constants of the
method (thresholds, rates, time values) are expected in the list.

Usage: python figures/audit_numbers.py manuscript|paperB
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pool = set()


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, float):
        for x in (o, abs(o), 1 - o):
            for nd in (2, 3, 4):
                pool.add(f"{x:.{nd}f}")


for f in (ROOT / "outputs").rglob("*.json"):
    walk(json.loads(f.read_text()))
miss = {}
for f in sorted((ROOT / sys.argv[1]).rglob("*.tex")):
    for n in re.findall(r"(?<![\w.])\d\.\d{3,4}(?!\d)", f.read_text()):
        if n not in pool:
            miss.setdefault(n, set()).add(f.name)
for n, fs in sorted(miss.items()):
    print(n, sorted(fs))
print(f"{len(miss)} numbers without a match in outputs/")
