# Harness directory

Repo-local harness configuration and state.

```text
harness/
  policies/                  standing research and manuscript rules
  task_specs/                declarative task definitions
  memory/                    the durable project memory journal (versioned)
  reports/                   generated run reports and history (ignored)
  papercheck.toml            engine configuration
  wall_status.example.toml   example Wall drafting-order status
  forbidden_phrases.txt      project-specific banned phrases
  table_manifest.example.yaml example table-value manifest
```

## Policies

`policies/` holds the standing rules: `claim_policy`, `rigor_policy`,
`narrative_policy`, `submission_policy`, `privacy_policy`,
`amplification_policy`, and `memory_policy`. They are the human-readable
source of the rules the engine enforces.

## Configuration flow

`papercheck.toml` configures the engine. `venue.blinding` and
`venue.human_subjects` drive the compliance checks; the `title`,
`abstract`, and `keywords` bounds tune the front-matter checks;
`checks.disable` and `checks.warn_only` retune strictness. The CLI reads
`harness/papercheck.toml` by default and merges it over the built-in
defaults.

## Memory and reports

`memory/memory.jsonl` is the project's durable knowledge and is committed
to git. `reports/` holds generated run reports, the papercheck run
history (`history.jsonl`), and the improvement report; it is ignored by
git.

Run verification with:

```bash
make -f makefiles/ManuscriptHarness.mk harness-verify
```
