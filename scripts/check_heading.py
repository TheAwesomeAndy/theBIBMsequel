#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re
import sys
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('manuscript/main.tex')
if not p.exists():
    print('missing file')
    raise SystemExit(1)
text = p.read_text(encoding='utf-8', errors='replace')
m = re.search(r'\\title\{([^{}]*)\}', text, flags=re.S)
if not m:
    print('missing title')
    raise SystemExit(1)
title = m.group(1).strip()
count = len(re.findall(r'\b\w+\b', title))
if count < 8 or count > 13 or '?' in title:
    print(f'bad title: {count} words')
    raise SystemExit(1)
print('title check passed')
