#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('manuscript/main.tex')
if not p.exists():
    print('missing file')
    raise SystemExit(1)
text = p.read_text(encoding='utf-8', errors='replace').lower()
missing = []
for term in ['baseline', 'limitation', 'ethics', 'availability']:
    if term not in text:
        missing.append(term)
if missing:
    print('missing signals: ' + ', '.join(missing))
    raise SystemExit(1)
print('signal check passed')
