#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('manuscript/main.tex')
if not p.exists():
    print('missing file')
    raise SystemExit(1)
text = p.read_text(encoding='utf-8', errors='replace').lower()
fail = []
if 'the rest of this paper is organized as follows' in text:
    fail.append('roadmap filler')
if 'proves that' in text:
    fail.append('overstrong claim')
if 'clearly proves' in text:
    fail.append('overstrong claim')
if fail:
    print('quality gate failed')
    print('\n'.join(fail))
    raise SystemExit(1)
print('quality gate passed')
