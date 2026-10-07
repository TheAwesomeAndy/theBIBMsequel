#!/usr/bin/env python3
"""Order a manual thebibliography by first citation (IEEE numbering) and check coverage.

Usage: python scripts/order_bibliography.py manuscript/journal/main_journal.tex [--write]
Reports cited keys without a \\bibitem and bibitems that are never cited. With --write,
rewrites the bibliography in order of first citation and drops uncited items.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    s = path.read_text(encoding="utf-8")
    a = s.index("\\begin{thebibliography}")
    b = s.index("\\end{thebibliography}")
    head = s[a:s.index("\n", a) + 1]
    body = s[s.index("\n", a) + 1:b]
    text = re.sub(r"(?<!\\)%.*", "", s[:a])
    order = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    items = {}
    for blk in re.split(r"(?=\\bibitem\{)", body):
        m = re.match(r"\\bibitem\{([^}]*)\}", blk)
        if m:
            items[m.group(1)] = blk.strip() + "\n"
    missing = [k for k in order if k not in items]
    uncited = [k for k in items if k not in order]
    print(f"[bib] cited {len(order)}, items {len(items)}, missing {missing}, uncited {uncited}")
    if "--write" in sys.argv:
        new = head + "\n" + "\n".join(items[k] for k in order if k in items) + "\n"
        path.write_text(s[:a] + new + s[b:], encoding="utf-8")
        print(f"[bib] rewrote {path} in citation order")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
