"""Scripted policy schedules (§10 Stage 10a) — the single source of truth for the
static policy grid, shared by the DGP's Regime-F ground truth, the ABM, and the RL baselines."""

from __future__ import annotations

from regworld.rules import PolicyLevers
from regworld.types import PolicyCfg

STATIC_POLICIES: dict[str, PolicyLevers] = {
    "none": PolicyLevers(0.0, 0.0, 0.0, 0.0),
    "uniform_low": PolicyLevers(0.3, 0.0, 0.5, 0.0),
    "uniform_high": PolicyLevers(0.9, 0.0, 0.9, 0.0),
    "targeted": PolicyLevers(0.6, 0.8, 0.7, 0.0),
    "phased_targeted": PolicyLevers(0.6, 0.5, 0.3, 0.3),
}


def levers_from_config(policy: PolicyCfg) -> PolicyLevers:
    if policy.kind == "static":
        if policy.name in STATIC_POLICIES:
            return STATIC_POLICIES[policy.name]
        return PolicyLevers(
            policy.enforcement, policy.targeting, policy.phase_speed, policy.subsidy
        )
    raise ValueError(f"policy {policy.name!r} is learned; load its artifact instead")
