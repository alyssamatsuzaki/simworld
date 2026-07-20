"""Gymnasium interface over the interpretable Mesa regulation model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from regworld.rules import (
    Constants,
    FirmAttributes,
    PolicyLevers,
    QuarterOutcome,
    SegmentAttributes,
    WorldState,
    hhi,
    regulator_reward,
)
from regworld.types import RegWorldConfig

from .wrappers import (
    build_flat_observation,
    flat_observation_space,
    regulator_action_space,
)


class RegulationBackend(Protocol):
    state: WorldState
    firms: FirmAttributes
    segments: SegmentAttributes
    baseline_outcome: QuarterOutcome
    last_outcome: QuarterOutcome
    last_firm_rewards: Mapping[int, float] | NDArray[np.float64]

    def step_with_controls(
        self,
        policy: PolicyLevers | None = None,
        strategic_actions: Mapping[int, NDArray[np.float64]] | None = None,
        controls: object | None = None,
    ) -> object: ...


ModelFactory = Callable[[RegWorldConfig, int], RegulationBackend]


def _default_model_factory(cfg: RegWorldConfig, seed: int) -> RegulationBackend:
    from regworld.abm.model import RegulationModel, load_observed_world

    return cast(
        RegulationBackend,
        RegulationModel(cfg, world=load_observed_world(cfg, seed=seed), seed=seed),
    )


def _outcome_from_state(model: RegulationBackend) -> QuarterOutcome:
    state, firms, segments = model.state, model.firms, model.segments
    alive = state.alive
    revenue = state.revenue * alive
    by_tercile: list[float] = []
    for tercile in range(3):
        mask = alive & (firms.size_tercile == tercile)
        by_tercile.append(float(np.mean(state.y[mask])) if np.any(mask) else 0.0)
    return QuarterOutcome(
        float(np.mean(state.y[alive])) if np.any(alive) else 0.0,
        float(np.sum(state.y * revenue) / max(float(np.sum(revenue)), 1e-9)),
        (by_tercile[0], by_tercile[1], by_tercile[2]),
        hhi(state.revenue, alive),
        float(np.sum(segments.weight * state.trust) / np.sum(segments.weight)),
        0.0,
        float(1.0 - np.sum(alive) / firms.n),
        0.0,
        0,
    )


def _model_baseline(model: RegulationBackend) -> QuarterOutcome:
    value = getattr(model, "baseline_outcome", None)
    return value if isinstance(value, QuarterOutcome) else _outcome_from_state(model)


def _model_outcome(model: RegulationBackend, result: object) -> QuarterOutcome:
    if isinstance(result, QuarterOutcome):
        return result
    if isinstance(result, tuple):
        for value in result:
            if isinstance(value, QuarterOutcome):
                return value
    value = getattr(model, "last_outcome", None)
    if isinstance(value, QuarterOutcome):
        return value
    raise RuntimeError("RegulationModel.step must return or expose a QuarterOutcome")


def _step_backend(
    model: RegulationBackend,
    policy: PolicyLevers,
    strategic_actions: Mapping[int, NDArray[np.float64]] | None = None,
) -> QuarterOutcome:
    result = model.step_with_controls(policy=policy, strategic_actions=strategic_actions)
    return _model_outcome(model, result)


class AbmEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(self, cfg: RegWorldConfig, *, model_factory: ModelFactory | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_space = regulator_action_space()
        self.observation_space = flat_observation_space(cfg)
        self._model_factory = model_factory or _default_model_factory
        self.model: RegulationBackend | None = None
        self._baseline: QuarterOutcome | None = None
        self._outcome: QuarterOutcome | None = None
        self._elapsed = 0
        self._cumulative_audits = 0
        self._last_action = np.zeros(4, dtype=np.float32)

    def _observation(self) -> NDArray[np.float32]:
        if self.model is None or self._baseline is None or self._outcome is None:
            raise RuntimeError("reset() must be called before observing")
        return build_flat_observation(
            self.cfg,
            self.model.state,
            self.model.firms,
            self._outcome,
            self._baseline,
            self._last_action,
            self._elapsed,
            self._cumulative_audits,
        )

    def _budget_remaining(self) -> float:
        if self.model is None:
            return 1.0
        constants = cast(Constants, getattr(self.model, "constants", Constants()))
        capacity = max(self.cfg.horizon_quarters * constants.audit_budget * self.model.firms.n, 1.0)
        return max(0.0, 1.0 - self._cumulative_audits / capacity)

    def _collapsed(self) -> bool:
        if self._outcome is None:
            return False
        return bool(
            self._outcome.exit_rate_cum > 0.40
            or (
                self._elapsed > 12
                and self._outcome.compliance_rate < 0.05
                and self._budget_remaining() <= 0.0
            )
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        model_seed = self.cfg.seed if seed is None else seed
        self.model = self._model_factory(self.cfg, model_seed)
        self._baseline = _model_baseline(self.model)
        self._outcome = self._baseline
        self._elapsed = self._cumulative_audits = 0
        self._last_action = np.zeros(4, dtype=np.float32)
        return self._observation(), {"seed": model_seed}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.model is None or self._baseline is None:
            raise RuntimeError("reset() must be called before step()")
        action_box = cast(gym.spaces.Box, self.action_space)
        self._last_action = np.clip(action, action_box.low, action_box.high).astype(
            np.float32, copy=False
        )
        policy = PolicyLevers(*[float(value) for value in self._last_action])
        self._outcome = _step_backend(self.model, policy)
        self._elapsed += 1
        self._cumulative_audits += self._outcome.n_audits
        weights = tuple(
            float(getattr(self.cfg.objective, name))
            for name in ("w_c", "w_h", "w_s", "w_e", "w_t", "w_x")
        )
        constants = cast(Constants, getattr(self.model, "constants", Constants()))
        reward = regulator_reward(
            self._outcome,
            self._baseline,
            cast(tuple[float, float, float, float, float, float], weights),
            constants,
            self.model.firms.n,
        )
        terminated = self._collapsed()
        truncated = self._elapsed >= self.cfg.horizon_quarters and not terminated
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {"outcome": self._outcome, "elapsed_quarters": self._elapsed},
        )
