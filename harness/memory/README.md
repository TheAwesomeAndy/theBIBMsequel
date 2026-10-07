# Project memory (Module 7)

Durable, inspectable project knowledge. Unlike an AI context window, this
journal survives context loss, is versioned in git, and can be reviewed
by a human.

## Journal format

`memory.jsonl` is one JSON object per line:

```json
{"id": 1, "timestamp": "...", "kind": "decision", "text": "...", "tags": ["..."], "status": "open"}
```

Kinds:

- `decision` - an architectural or framing choice, with its rationale.
- `lesson` - what a gate (or a human) caught and how it was fixed.
- `reviewer` - external review feedback, recorded verbatim.
- `question` - an open question to resolve before submission.

Status is `open` or `resolved`; a resolved entry carries a `resolution`.

## Workflow

1. At session start, read the digest:
   `python scripts/memory.py digest`.
2. Record entries at every phase boundary and whenever a gate fires or a
   human corrects the work:
   `python scripts/memory.py add lesson "papercheck caught X; fixed by Y" --tags methods`.
3. Resolve entries when acted on:
   `python scripts/memory.py resolve 3 "Added Transformer baseline in Section 4.2"`.
4. After each drafting round and each review round, run the analyzer:
   `python scripts/self_improve.py`.

## Closing the loop (Rule 7.5)

`self_improve.py` mines `harness/reports/history.jsonl` (written by
`papercheck --history`) and this journal for recurring findings, trends,
chronic failures, and reviewer echoes, then suggests concrete harness
upgrades. A mistake that reaches a reviewer twice is a harness bug:
encode it as a deterministic check or a standing rule so it cannot recur
silently.

## Privacy

Do not store personal reviewer identities, private contact details, or
restricted data in the journal. Record the substance of feedback, not
who gave it.
