"""PettingZoo Parallel environment for a regulator and strategic large firms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from regworld.rules import Constants, PolicyLevers, QuarterOutcome, regulator_reward
from regworld.types import RegWorldConfig

from .abm_env import (
    ModelFactory,
    RegulationBackend,
    _default_model_factory,
    _model_baseline,
    _step_backend,
)
from .wrappers import (
    build_flat_observation,
    flat_observation_space,
    regulator_action_space,
)

FirmObservation = dict[str, NDArray[np.float32]]
AgentObservation = NDArray[np.float32] | FirmObservation


class RegulationMARLEnv(ParallelEnv):
    """Hybrid MARL world whose firm actions change the actual ABM transition."""

    metadata: ClassVar[dict[str, Any]] = {
        "name": "regworld_parallel_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(self, cfg: RegWorldConfig, *, model_factory: ModelFactory | None = None) -> None:
        self.cfg = cfg
        self._model_factory = model_factory or _default_model_factory
        self.n_strategic = cfg.env.n_strategic_firms
        self.possible_agents = ["regulator_0"] + [f"firm_{i}" for i in range(self.n_strategic)]
        self.agents: list[str] = []
        self.action_spaces: dict[str, spaces.Space[Any]] = {"regulator_0": regulator_action_space()}
        self.action_spaces.update(
            {
                f"firm_{i}": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
                for i in range(self.n_strategic)
            }
        )
        self.observation_spaces: dict[str, spaces.Space[Any]] = {
            "regulator_0": flat_observation_space(cfg)
        }
        self.observation_spaces.update(
            {
                f"firm_{i}": spaces.Dict(
                    {
                        "global": flat_observation_space(cfg),
                        "local": spaces.Box(0.0, 1.0, shape=(15,), dtype=np.float32),
                    }
                )
                for i in range(self.n_strategic)
            }
        )
        self.model: RegulationBackend | None = None
        self._baseline: QuarterOutcome | None = None
        self._outcome: QuarterOutcome | None = None
        self._strategic_ids: dict[str, int] = {}
        self._last_regulator_action = np.zeros(4, dtype=np.float32)
        self._last_firm_actions: dict[str, NDArray[np.float32]] = {}
        self._elapsed = self._cumulative_audits = 0

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def _select_firms(self) -> dict[str, int]:
        if self.model is None:
            raise RuntimeError("reset() must be called first")
        alive = np.flatnonzero(self.model.state.alive)
        if alive.size < self.n_strategic:
            raise RuntimeError(f"need {self.n_strategic} living firms, found {alive.size}")
        chosen = alive[np.argsort(-self.model.firms.size[alive])[: self.n_strategic]]
        return {f"firm_{rank}": int(firm_id) for rank, firm_id in enumerate(chosen)}

    def _global(self) -> NDArray[np.float32]:
        if self.model is None or self._baseline is None or self._outcome is None:
            raise RuntimeError("reset() must be called first")
        return build_flat_observation(
            self.cfg,
            self.model.state,
            self.model.firms,
            self._outcome,
            self._baseline,
            self._last_regulator_action,
            self._elapsed,
            self._cumulative_audits,
        )

    def _local(self, agent: str) -> NDArray[np.float32]:
        if self.model is None:
            raise RuntimeError("reset() must be called first")
        firm_id = self._strategic_ids[agent]
        state, firms = self.model.state, self.model.firms
        total_revenue = max(float(np.sum(state.revenue * state.alive)), 1e-9)
        association = firms.association[firm_id]
        mask = (firms.association == association) & state.alive
        assoc_compliance = float(np.mean(state.y[mask])) if np.any(mask) else 0.0
        action = self._last_firm_actions.get(agent, np.zeros(3, dtype=np.float32))
        phase_length = 12.0 - 10.0 * float(self._last_regulator_action[2])
        values = np.array(
            [
                state.y[firm_id],
                float(state.alive[firm_id]),
                state.revenue[firm_id] / total_revenue,
                np.mean(firms.size <= firms.size[firm_id]),
                firms.data_intensity[firm_id],
                min(float(firms.cost_coef[firm_id]) / 5.0, 1.0),
                float(state.audited[firm_id]),
                min(
                    float(state.fines[firm_id]) / max(float(state.revenue[firm_id]), 1e-9),
                    1.0,
                ),
                min(float(state.tenure[firm_id]) / 12.0, 1.0),
                assoc_compliance,
                self._last_regulator_action[0],
                min(1.0, self._elapsed / phase_length),
                action[0],
                action[1],
                action[2],
            ],
            dtype=np.float32,
        )
        return np.clip(values, 0.0, 1.0)

    def _observations(self, agents: list[str] | None = None) -> dict[str, AgentObservation]:
        global_obs = self._global()
        observed_agents = self.agents if agents is None else agents
        return {
            agent: global_obs.copy()
            if agent == "regulator_0"
            else {"global": global_obs.copy(), "local": self._local(agent)}
            for agent in observed_agents
        }

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
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, AgentObservation], dict[str, dict[str, Any]]]:
        del options
        model_seed = self.cfg.seed if seed is None else seed
        self.model = self._model_factory(self.cfg, model_seed)
        self._baseline = _model_baseline(self.model)
        self._outcome = self._baseline
        self._strategic_ids = self._select_firms()
        self.agents = self.possible_agents.copy()
        self._last_regulator_action = np.zeros(4, dtype=np.float32)
        self._last_firm_actions = {
            agent: np.zeros(3, dtype=np.float32) for agent in self._strategic_ids
        }
        self._elapsed = self._cumulative_audits = 0
        infos = {
            agent: (
                {"seed": model_seed}
                if agent == "regulator_0"
                else {"firm_id": self._strategic_ids[agent]}
            )
            for agent in self.agents
        }
        return self._observations(), infos

    def step(
        self, actions: dict[str, NDArray[np.float32]]
    ) -> tuple[
        dict[str, AgentObservation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self.model is None or self._baseline is None:
            raise RuntimeError("reset() must be called before step()")
        agents_before = self.agents.copy()
        regulator_box = cast(spaces.Box, self.action_spaces["regulator_0"])
        regulator_action = np.clip(actions["regulator_0"], regulator_box.low, regulator_box.high)
        self._last_regulator_action = regulator_action.astype(np.float32, copy=False)
        policy = PolicyLevers(*[float(value) for value in regulator_action])
        strategic: dict[int, NDArray[np.float64]] = {}
        for agent in agents_before:
            if agent != "regulator_0":
                action = np.clip(actions[agent], 0.0, 1.0).astype(np.float32, copy=False)
                self._last_firm_actions[agent] = action
                strategic[self._strategic_ids[agent]] = action.astype(np.float64)
        self._outcome = _step_backend(self.model, policy, strategic)
        self._elapsed += 1
        self._cumulative_audits += self._outcome.n_audits
        weights = cast(
            tuple[float, float, float, float, float, float],
            tuple(
                float(getattr(self.cfg.objective, name))
                for name in ("w_c", "w_h", "w_s", "w_e", "w_t", "w_x")
            ),
        )
        constants = cast(Constants, getattr(self.model, "constants", Constants()))
        rewards = {
            "regulator_0": float(
                regulator_reward(
                    self._outcome,
                    self._baseline,
                    weights,
                    constants,
                    self.model.firms.n,
                )
            )
        }
        profits = self.model.last_firm_rewards
        if not isinstance(profits, Mapping) and not isinstance(profits, np.ndarray):
            raise RuntimeError("last_firm_rewards must index realized profit by firm_id")
        for agent in agents_before:
            if agent != "regulator_0":
                firm_id = self._strategic_ids[agent]
                if isinstance(profits, Mapping) and firm_id not in profits:
                    raise RuntimeError(f"missing realized profit for strategic firm {firm_id}")
                if isinstance(profits, np.ndarray) and not 0 <= firm_id < profits.size:
                    raise RuntimeError(f"missing realized profit for strategic firm {firm_id}")
                rewards[agent] = float(profits[firm_id])
        collapsed = self._collapsed()
        timed_out = self._elapsed >= self.cfg.horizon_quarters
        terminations = {
            agent: bool(
                collapsed
                or (
                    agent != "regulator_0"
                    and not self.model.state.alive[self._strategic_ids[agent]]
                )
            )
            for agent in agents_before
        }
        truncations = {
            agent: bool(timed_out and not terminations[agent]) for agent in agents_before
        }
        observations = self._observations(agents_before)
        self.agents = [
            agent for agent in agents_before if not terminations[agent] and not truncations[agent]
        ]
        infos = {
            agent: (
                {"elapsed_quarters": self._elapsed}
                if agent == "regulator_0"
                else {
                    "firm_id": self._strategic_ids[agent],
                    "elapsed_quarters": self._elapsed,
                }
            )
            for agent in agents_before
        }
        return observations, rewards, terminations, truncations, infos

    def state(self) -> NDArray[np.float32]:
        return self._global()

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []
