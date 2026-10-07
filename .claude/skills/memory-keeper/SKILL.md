---
name: memory-keeper
description: Read, write, and resolve the project memory journal and run the self-improvement retrospective (Module 7). Use at session start, phase boundaries, and after reviews.
---

# Memory Keeper

Use to keep the project's durable knowledge current.

## When to use
- Session start: `python scripts/memory.py digest` and load it.
- Phase boundary or correction: add a `decision` or `lesson` entry.
- Review round: add `reviewer` entries; run the retrospective.

## Procedure
1. Read the digest before acting.
2. `python scripts/memory.py add <kind> "<text>" --tags <a,b>` for new
   knowledge; `resolve <id> "<resolution>"` when acted on.
3. `python scripts/self_improve.py` and act on its suggestions.

## Verification
Open entries reflect reality; recurring findings have a proposed harness
upgrade; no private reviewer identities are stored.

## Stop point
Present the digest or the improvement report and stop.
