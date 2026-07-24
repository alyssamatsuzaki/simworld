"""Gymnasium interface over the interpretable Mesa regulation model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from simworld.rules import (
    Constants,
    FirmAttributes,
    PolicyLevers,
    QuarterOutcome,
    SegmentAttributes,
    Theta,
    WorldState,
    hhi,
    regulator_reward,
)
from simworld.types import SimWorldConfig

from .wrappers import (
    build_flat_observation,
    flat_observation_space,
    regulator_action_space,
)

log = logging.getLogger(__name__)

# Fraction of one quarter's maximum audit budget below which the remaining
# horizon budget counts as exhausted for the collapse test (§10 Stage 8): the
# regulator cannot fund even a token 5%-of-normal quarter again.
BUDGET_EXHAUSTED_QUARTER_FRACTION = 0.05


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


ModelFactory = Callable[[SimWorldConfig, int], RegulationBackend]

# Posterior-mean thetas cached per (posterior path, mtime): one arviz load per
# artifact, not one per env reset. The warn-set keeps the prior-center fallback
# to a single WARNING per missing artifact path.
_THETA_CACHE: dict[tuple[str, int], Theta] = {}
_WARNED_PRIOR_CENTER: set[str] = set()


def _load_posterior_mean_theta(path: Path) -> Theta:
    from simworld.calibration.posterior import posterior_mean_theta

    return posterior_mean_theta(path)


def _default_theta(cfg: SimWorldConfig) -> Theta | None:
    """Posterior-mean Theta when Stage 4 has calibrated one, else None.

    PLAN Stage 3: the ABM behind the env is "parameterized by *estimated*
    values" — the oracle that grades RL policies and the exploitation gap must
    run at the calibrated parameters the emulator was trained around, falling
    back to prior-center ``Theta()`` (with one warning) before calibration ran.
    """
    from simworld.calibration.posterior import posterior_path

    path = posterior_path(cfg)
    key = str(path)
    if not path.is_file():
        if key not in _WARNED_PRIOR_CENTER:
            _WARNED_PRIOR_CENTER.add(key)
            log.warning(
                "calibrated posterior missing at %s; env oracle falls back to "
                "prior-center Theta() (run `make calibrate` for estimated values)",
                path,
            )
        return None
    cache_key = (key, path.stat().st_mtime_ns)
    if cache_key not in _THETA_CACHE:
        _THETA_CACHE[cache_key] = _load_posterior_mean_theta(path)
    return _THETA_CACHE[cache_key]


def _default_model_factory(cfg: SimWorldConfig, seed: int) -> RegulationBackend:
    from simworld.abm.model import RegulationModel, load_observed_world

    return cast(
        RegulationBackend,
        RegulationModel(
            cfg,
            world=load_observed_world(cfg, seed=seed),
            theta=_default_theta(cfg),
            seed=seed,
        ),
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

    def __init__(self, cfg: SimWorldConfig, *, model_factory: ModelFactory | None = None) -> None:
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

    def _budget_exhausted(self) -> bool:
        """Remaining horizon budget below 5% of one quarter's maximum audits."""
        remaining_quarters = self._budget_remaining() * self.cfg.horizon_quarters
        return remaining_quarters < BUDGET_EXHAUSTED_QUARTER_FRACTION

    def _collapsed(self) -> bool:
        if self._outcome is None:
            return False
        return bool(
            self._outcome.exit_rate_cum > 0.40
            or (
                self._elapsed > 12
                and self._outcome.compliance_rate < 0.05
                and self._budget_exhausted()
            )
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Gymnasium convention: seed=None continues the env RNG stream (fresh
        # episode noise per auto-reset); an explicit seed re-pins it so
        # reset(seed=k) twice replays the identical episode.
        super().reset(seed=seed)
        del options
        model_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
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
