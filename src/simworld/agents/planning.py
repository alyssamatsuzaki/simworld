"""Roll a :data:`~simworld.agents.registry.PolicyFn` in either world and
compare — the planning-utility acid test needs both numbers on the same
episode-return scale.

``evaluate_in_abm`` runs the true Mesa simulator; ``evaluate_in_emulator``
runs the trained ``WorldModel`` in imagination. Both return a
:class:`RolloutStats` averaged over ``seeds x draws`` so the caller gets a
mean and a 95% CI cheaply, with the grid size controlled entirely by the
caller (kept small at the smoke profile).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from simworld.agents.registry import PolicyFn
from simworld.environments.abm_env import AbmEnv, ModelFactory
from simworld.environments.emulator_env import EmulatorEnv
from simworld.models.world_model import WorldModel
from simworld.types import SimWorldConfig


@dataclass(frozen=True)
class RolloutStats:
    mean: float
    std: float
    n: int
    returns: tuple[float, ...]

    def ci95(self) -> tuple[float, float]:
        """Normal-approximation 95% CI on the mean (n=1 collapses to +/-0)."""
        se = self.std / max(np.sqrt(self.n), 1.0)
        return self.mean - 1.96 * se, self.mean + 1.96 * se


def rollout_episode(
    env: gym.Env[NDArray[np.float32], NDArray[np.float32]], policy: PolicyFn, seed: int
) -> float:
    """One full episode under ``policy``; returns the summed reward J."""
    observation, _info = env.reset(seed=seed)
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        action = policy(observation)
        observation, reward, terminated, truncated, _info = env.step(action)
        total += float(reward)
    return total


def _seed_grid(seeds: Sequence[int], draws: int) -> list[int]:
    """Deterministically expand each base seed into ``draws`` distinct rollout seeds."""
    grid: list[int] = []
    for base_seed in seeds:
        rng = np.random.default_rng(base_seed)
        grid.extend(int(value) for value in rng.integers(0, 2**31 - 1, size=draws))
    return grid


def _stats(returns: list[float]) -> RolloutStats:
    arr = np.asarray(returns, dtype=np.float64)
    return RolloutStats(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n=len(returns),
        returns=tuple(float(v) for v in returns),
    )


def evaluate_in_abm(
    cfg: SimWorldConfig,
    policy: PolicyFn,
    seeds: Sequence[int],
    *,
    draws: int = 1,
    model_factory: ModelFactory | None = None,
) -> RolloutStats:
    """Roll ``policy`` in the true ABM across ``seeds x draws`` episodes."""
    env = AbmEnv(cfg, model_factory=model_factory)
    try:
        returns = [rollout_episode(env, policy, seed) for seed in _seed_grid(seeds, draws)]
    finally:
        env.close()
    return _stats(returns)


def evaluate_in_emulator(
    cfg: SimWorldConfig,
    policy: PolicyFn,
    model: WorldModel,
    meta: dict[str, Any],
    seeds: Sequence[int],
    *,
    draws: int = 1,
) -> RolloutStats:
    """Roll ``policy`` in imagination (``EmulatorEnv``) across ``seeds x draws`` episodes."""
    env = EmulatorEnv(cfg, model=model, meta=meta)
    try:
        returns = [rollout_episode(env, policy, seed) for seed in _seed_grid(seeds, draws)]
    finally:
        env.close()
    return _stats(returns)


def exploitation_gap(j_emulator: float, j_abm: float) -> float:
    """(J_emulator - J_ABM) / |J_ABM|: a policy that looks great in the model
    and mediocre in the true ABM has steered into the model's errors."""
    denom = max(abs(j_abm), 1e-6)
    return (j_emulator - j_abm) / denom
