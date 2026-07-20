"""Shared Gymnasium spaces, observation encoding, and lightweight wrappers."""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from regworld.rules import Constants, FirmAttributes, QuarterOutcome, WorldState
from regworld.types import RegWorldConfig

ACTION_LOW = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.ones(4, dtype=np.float32)


def regulator_action_space() -> spaces.Box:
    return spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)


def flat_observation_space(cfg: RegWorldConfig) -> spaces.Box:
    n = 25 + cfg.population.n_sectors
    low = np.zeros(n, dtype=np.float32)
    high = np.ones(n, dtype=np.float32)
    low[2], high[2] = 0.0, 10_000.0
    low[3], high[3] = -10_000.0, 10_000.0
    low[5], high[5] = -10.0, 10.0
    action_start = 21 + cfg.population.n_sectors
    low[action_start:] = ACTION_LOW
    high[action_start:] = ACTION_HIGH
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _group_compliance(
    state: WorldState, groups: NDArray[np.int64], n_groups: int
) -> NDArray[np.float32]:
    values = np.zeros(n_groups, dtype=np.float32)
    for group in range(n_groups):
        mask = state.alive & (groups == group)
        if np.any(mask):
            values[group] = float(np.mean(state.y[mask]))
    return values


def _size_deciles(size: NDArray[np.float64]) -> NDArray[np.int64]:
    if size.size <= 1:
        return np.zeros(size.size, dtype=np.int64)
    return np.digitize(size, np.quantile(size, np.linspace(0.1, 0.9, 9))).astype(np.int64)


def build_flat_observation(
    cfg: RegWorldConfig,
    state: WorldState,
    firms: FirmAttributes,
    outcome: QuarterOutcome,
    baseline: QuarterOutcome,
    last_action: NDArray[np.float32],
    elapsed: int,
    cumulative_audits: int,
) -> NDArray[np.float32]:
    alive_count = max(int(np.sum(state.alive)), 1)
    max_audits = max(cfg.horizon_quarters * Constants().audit_budget * firms.n, 1.0)
    budget_remaining = 1.0 - cumulative_audits / max_audits
    cs_scale = max(abs(baseline.consumer_surplus), 1e-9)
    cs_index = (outcome.consumer_surplus - baseline.consumer_surplus) / cs_scale
    revenue_total = max(float(np.sum(state.revenue * state.alive)), 1e-9)
    penalty_rate = float(np.sum(state.fines)) / revenue_total
    sectors = _group_compliance(state, firms.sector, cfg.population.n_sectors)
    deciles = _group_compliance(state, _size_deciles(firms.size), 10)
    obs = np.concatenate(
        [
            np.array(
                [
                    outcome.compliance_rate,
                    outcome.compliance_rate_weighted,
                    outcome.hhi,
                    outcome.hhi - baseline.hhi,
                    outcome.mean_trust,
                    np.clip(cs_index, -10.0, 10.0),
                    outcome.exit_rate_cum,
                    np.clip(budget_remaining, 0.0, 1.0),
                    np.clip(elapsed / max(cfg.horizon_quarters, 1), 0.0, 1.0),
                ],
                dtype=np.float32,
            ),
            sectors,
            deciles,
            np.array(
                [outcome.n_audits / alive_count, np.clip(penalty_rate, 0.0, 1.0)],
                dtype=np.float32,
            ),
            last_action,
        ]
    )
    space = flat_observation_space(cfg)
    return cast(
        NDArray[np.float32],
        np.clip(obs, space.low, space.high).astype(np.float32, copy=False),
    )


class ClipPolicyAction(
    gym.ActionWrapper[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
):
    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32, copy=False)
