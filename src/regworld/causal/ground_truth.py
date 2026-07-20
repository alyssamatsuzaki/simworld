"""The answer key: true intervention effects sealed by the DGP (§7.10, §1 firewall).

This module is one of the few allowed to read ``artifacts/oracle`` (see
``data/store.py``). Every Stage-5 estimator is graded against the numbers here;
nothing else in ``causal/`` may open the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from regworld.data.store import read_oracle
from regworld.types import RegWorldConfig


@dataclass(frozen=True)
class GroundTruthEffects:
    """The sealed truth the pipeline is graded against.

    ``onset_att`` is the total-regulation-onset ATT (enforcement e_low->e_high plus
    the phase and compliance-cost activation that ride with it), matching the DiD
    estimand exactly (DEVIATIONS 2026-07-20). ``audit_ate`` / ``audit_cate_by_tercile``
    are the analytic per-firm do(audited) effects along the observed trajectory.
    """

    onset_att: float
    onset_att_se: float
    onset_per_quarter: list[float]
    terminal: float
    audit_ate: float
    audit_cate_by_tercile: list[float]
    did_truth: float
    did_truth_se: float
    interference_gap: float
    n_do_seeds: int
    enforcement_low: float
    enforcement_high: float


def load_ground_truth(cfg: RegWorldConfig) -> GroundTruthEffects:
    """Read ``oracle/true_effects.json`` and expose it as a typed record."""
    raw = read_oracle(cfg, "true_effects")
    return GroundTruthEffects(
        onset_att=float(raw["tau_true_onset_att"]),
        onset_att_se=float(raw["tau_true_onset_att_se"]),
        onset_per_quarter=[float(x) for x in raw["tau_true_onset_per_quarter"]],
        terminal=float(raw["tau_true_terminal"]),
        audit_ate=float(raw["tau_true_audit_ate"]),
        audit_cate_by_tercile=[float(x) for x in raw["tau_true_audit_cate_by_tercile"]],
        did_truth=float(raw["tau_did_truth"]),
        did_truth_se=float(raw["tau_did_truth_se"]),
        interference_gap=float(raw["tau_interference_gap"]),
        n_do_seeds=int(raw["n_do_seeds"]),
        enforcement_low=float(raw["regulation_off_enforcement"]),
        enforcement_high=float(raw["regulation_on_enforcement"]),
    )
