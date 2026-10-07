#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

MARKERS = ['data_local', 'restricted_data']
EXTENSIONS = {'.md', '.tex', '.txt', '.py', '.yml', '.yaml', '.json'}
SKIP_PARTS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules'}
SKIP_FILES = {'README.md', 'tools/check_path_markers.py', 'harness/policies/privacy_policy.md'}


def iter_files(root: Path):
    skip_fixture_examples = root == Path('.')
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if skip_fixture_examples and path.parts and path.parts[0] == 'fixtures':
            continue
        normalized = path.as_posix()
        if normalized in SKIP_FILES:
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='.')
    args = parser.parse_args()

    root = Path(args.root)
    findings = []
    for path in iter_files(root):
        text = path.read_text(encoding='utf-8', errors='replace')
        for marker in MARKERS:
            pos = text.find(marker)
            if pos >= 0:
                line = text.count('\n', 0, pos) + 1
                findings.append(f'{path}:{line}: {marker}')

    if findings:
        print('path marker check failed')
        print('\n'.join(findings))
        return 1
    print('path marker check passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
