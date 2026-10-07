# SPDX-License-Identifier: Apache-2.0
"""papercheck: deterministic Q1-publication quality gates for LaTeX manuscripts.

This package is the all-in-one checker behind the research AI harness.
Every rule in the Q1 playbook that can be verified deterministically is
implemented here as a registered check. Rules that require judgment are
enforced by the agent playbook (templates/claude/) and documented in
docs/q1_rule_index.md.

Design principles:
- stdlib only, no third-party dependencies;
- every check has a stable dotted id (e.g. ``title.length``);
- checks never crash the run: exceptions become findings;
- severity is data, not code: profiles and config decide what fails.
"""

__version__ = "1.0.0"
