"""Stage 11 (Phase 6, §10): the Ray-scalable scenario ensemble.

Builds a ``(policy x posterior-draw x seed)`` scenario cube of terminal
outcomes from the trained emulator (:mod:`regworld.ensemble.cube`) and
cross-validates a stratified subsample against the true tensorized ABM
(:mod:`regworld.ensemble.validation`). This package uses only the observed-
world simulator and the trained emulator: it never imports ``regworld.dgp``
and never reads answer-key ground-truth artifacts (enforced by
``tests/test_no_dgp_leakage.py``).
"""

from __future__ import annotations

from regworld.ensemble.cube import EnsembleResult, build_cube, resolve_policies, run_ensemble
from regworld.ensemble.validation import ValidationReport, run_validation

__all__ = [
    "EnsembleResult",
    "ValidationReport",
    "build_cube",
    "resolve_policies",
    "run_ensemble",
    "run_validation",
]
