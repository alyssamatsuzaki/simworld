"""Thin wrapper binding theta* to the shared decision rules (§7.4).

The confounded and misspecified worlds include the latent ``beta_capacity * z``
term. The recovery-control world omits it so ``dgp=wellspecified`` is genuinely the
same likelihood Stage 4 fits. The ``interacted`` flag is exclusive to the separate
misspecified variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regworld.dgp.world import CONSTANTS, THETA_STAR
from regworld.rules import (
    FirmAttributes,
    Graphs,
    PolicyLevers,
    QuarterOutcome,
    SegmentAttributes,
    WorldState,
    initial_state,
    step_quarter,
)
from regworld.types import RegWorldConfig


@dataclass
class Trajectory:
    """One DGP run: per-quarter outcomes + the full firm-quarter covariate panel."""

    outcomes: list[QuarterOutcome]
    covariates: list[dict[str, np.ndarray]]  # one dict per quarter, arrays over firms
    final_state: WorldState

    def outcome_matrix(self) -> np.ndarray:
        """(T, 8) matrix in OUTCOME_VARIABLES order (reward/backfire filled downstream)."""
        return np.array(
            [
                [
                    o.compliance_rate,
                    o.compliance_rate_weighted,
                    o.hhi,
                    o.mean_trust,
                    o.consumer_surplus,
                    o.exit_rate_cum,
                    o.enforcement_cost,
                ]
                for o in self.outcomes
            ]
        )


def run_dgp(
    cfg: RegWorldConfig,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs: Graphs,
    levers: PolicyLevers,
    seed: int,
    quarters: int,
    *,
    t_start: np.ndarray | int = 0,
    start_state: WorldState | None = None,
) -> Trajectory:
    """Run the true world for `quarters` steps under `levers`.

    `t_start` may be a per-firm array (Regime P staggered rollout) or a scalar
    (Regime F: national onset). A large t_start (> quarters) means "never treated".
    """
    rng = np.random.default_rng(seed)
    state = (
        start_state
        if start_state is not None
        else initial_state(firms, segments, graphs, CONSTANTS, rng)
    )
    interacted = cfg.dgp.decision_rule == "logit_interacted"
    outcomes: list[QuarterOutcome] = []
    covariates: list[dict[str, np.ndarray]] = []
    for _ in range(quarters):
        state, outcome, covs = step_quarter(
            state,
            firms,
            segments,
            graphs,
            THETA_STAR,
            CONSTANTS,
            levers,
            rng,
            t_start=t_start,
            use_z=cfg.dgp.variant != "wellspecified",
            sticky=True,
            interacted=interacted,
        )
        outcomes.append(outcome)
        covariates.append(covs)
    return Trajectory(outcomes=outcomes, covariates=covariates, final_state=state)
