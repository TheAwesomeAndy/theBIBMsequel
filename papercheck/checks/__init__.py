# SPDX-License-Identifier: Apache-2.0
"""Importing this package registers every check module."""
from papercheck.checks import (  # noqa: F401
    bibliography,
    compliance,
    floats,
    front_matter,
    introduction,
    language,
    methods,
    results_discussion,
    structure,
)
