# Memory policy

The project memory journal (`harness/memory/memory.jsonl`) is durable,
versioned, human-readable knowledge.

- What belongs: decisions and their rationale, lessons a gate or a human
  taught, the substance of reviewer feedback, and open questions.
- Entry quality: each entry is a single, self-contained statement with
  tags. Vague entries are worthless later.
- Resolve discipline: mark an entry resolved when acted on, and record
  the resolution.
- Retrospective: run `scripts/self_improve.py` after each drafting round
  and review round; act on its suggestions.
- Close the loop (Rule 7.5): a mistake that reaches a reviewer twice
  becomes a deterministic check or a standing rule so it cannot recur
  silently.
- Privacy: never store personal reviewer identities, contact details, or
  restricted data. Record what was said, not who said it.
