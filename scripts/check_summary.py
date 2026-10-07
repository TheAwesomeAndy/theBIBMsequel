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
m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, flags=re.S | re.I)
if not m:
    print('missing abstract')
    raise SystemExit(1)
body = re.sub(r'\\[a-zA-Z]+(\{[^{}]*\})?', ' ', m.group(1))
words = len(re.findall(r'\b\w+\b', body))
sentences = len([s for s in re.split(r'[.!?]+\s*', body.strip()) if s.strip()])
if words > 200 or sentences != 4:
    print(f'bad abstract: {words} words, {sentences} sentences')
    raise SystemExit(1)
print('abstract check passed')
