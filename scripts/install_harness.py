#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Install the research AI harness into a target repository.

This script copies the harness directories, scripts, and makefiles from this
repository into a specified target directory. Use this after cloning the
harness repository to set up a project for AI-assisted manuscript auditing.

Example:
    python scripts/install_harness.py --target ../my-paper-repo
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SKIP_NAMES = {"__pycache__", ".pytest_cache"}


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists() or src.name in SKIP_NAMES:
        return
    if src.is_file():
        # A single file (e.g. project-instructions.md -> CLAUDE.md).
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if dst.exists():
        # Merge directories by copying individual files
        for item in src.iterdir():
            if item.name in SKIP_NAMES:
                continue
            target = dst / item.name
            if item.is_dir():
                copy_tree(item, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    else:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_NAMES))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=".",
        help="Target project directory (default: current directory)",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target_root = Path(args.target).resolve()
    if not target_root.exists():
        print(f"Creating target directory {target_root}")
        target_root.mkdir(parents=True)
    # Define relative paths to copy
    paths_to_copy = [
        ("templates/claude/.claude", ".claude"),
        ("templates/claude/project-instructions.md", "CLAUDE.md"),
        ("harness", "harness"),
        ("scripts", "scripts"),
        ("tools", "tools"),
        ("makefiles", "makefiles"),
        ("papercheck", "papercheck"),
        ("playbooks", "playbooks"),
    ]
    for src_rel, dest_rel in paths_to_copy:
        src_path = repo_root / src_rel
        dest_path = target_root / dest_rel
        copy_tree(src_path, dest_path)
    print(f"Harness installed into {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
