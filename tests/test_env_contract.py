"""Stage-8 Gymnasium contract tests using a deterministic lightweight backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray
from scipy import sparse

from regworld.abm.model import ObservedWorld, RegulationModel
from regworld.environments.abm_env import AbmEnv
from regworld.rules import (
    Constants,
    FirmAttributes,
    Graphs,
    PolicyLevers,
    QuarterOutcome,
    SegmentAttributes,
    Theta,
    WorldState,
    hhi,
)
from regworld.types import RegWorldConfig


class FakeRegulationModel:
    def __init__(
        self,
        cfg: RegWorldConfig,
        seed: int,
        *,
        collapse: bool = False,
        kill_largest: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.constants = Constants()
        self.collapse = collapse
        self.kill_largest = kill_largest
        n = max(cfg.env.n_strategic_firms + 2, 12)
        size = np.linspace(0.5, 2.5, n)
        self.firms = FirmAttributes(
            size,
            np.arange(n, dtype=np.int64) % cfg.population.n_sectors,
            np.linspace(0.2, 0.8, n),
            np.ones(n),
            np.zeros(n),
            np.full(n, 0.2),
            np.zeros(n),
            np.arange(n, dtype=np.int64) % cfg.population.n_associations,
            np.zeros(n, dtype=np.int64),
            np.digitize(size, np.quantile(size, [1 / 3, 2 / 3])).astype(np.int64),
        )
        self.segments = SegmentAttributes(
            np.array([1.0]), np.array([0.5]), np.array([float(n)]), np.array([0.6])
        )
        self.state = WorldState(
            np.zeros(n),
            np.ones(n, dtype=bool),
            size.copy(),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n, dtype=bool),
            size[None, :].copy(),
            np.array([0.6]),
            np.zeros(cfg.population.n_sectors),
            np.repeat(size[:, None], 3, axis=1),
            np.zeros(n),
            0,
        )
        self.last_firm_rewards = {i: float(size[i]) for i in range(n)}
        self.last_strategic_controls: dict[int, NDArray[np.float64]] = {}
        self._lobby = np.zeros(cfg.population.n_associations)
        self.baseline_outcome = self._outcome(0)
        self.last_outcome = self.baseline_outcome

    def _outcome(self, audits: int) -> QuarterOutcome:
        alive, revenue = self.state.alive, self.state.revenue * self.state.alive
        terciles = tuple(
            float(np.mean(self.state.y[alive & (self.firms.size_tercile == k)]))
            if np.any(alive & (self.firms.size_tercile == k))
            else 0.0
            for k in range(3)
        )
        return QuarterOutcome(
            float(np.mean(self.state.y[alive])) if np.any(alive) else 0.0,
            float(np.sum(self.state.y * revenue) / max(float(np.sum(revenue)), 1e-9)),
            terciles,
            hhi(self.state.revenue, alive),
            float(self.state.trust[0]),
            1.0,
            float(1.0 - np.mean(alive)),
            audits * self.constants.audit_unit_cost,
            audits,
        )

    def step_with_controls(
        self,
        policy: PolicyLevers | None = None,
        strategic_actions: Mapping[int, NDArray[np.float64]] | None = None,
        controls: object | None = None,
    ) -> None:
        del controls
        policy, strategic_actions = policy or PolicyLevers(), strategic_actions or {}
        old_lobby, next_lobby = self._lobby.copy(), np.zeros_like(self._lobby)
        y, audited, fines = (
            self.state.y.copy(),
            np.zeros(self.firms.n, bool),
            np.zeros(self.firms.n),
        )
        rewards: dict[int, float] = {}
        for firm_id in range(self.firms.n):
            invest, lobby, evade = strategic_actions.get(firm_id, np.zeros(3))
            assoc = self.firms.association[firm_id]
            enforcement = policy.enforcement * (1 - 0.5 * old_lobby[assoc])
            y[firm_id] = self.rng.random() < np.clip(0.1 + 0.7 * enforcement + 0.8 * invest, 0, 1)
            audited[firm_id] = self.rng.random() < 0.2 * enforcement
            detected = audited[firm_id] and not y[firm_id] and self.rng.random() > 0.8 * evade
            fines[firm_id] = 0.1 * self.state.revenue[firm_id] if detected else 0.0
            next_lobby[assoc] = max(next_lobby[assoc], lobby)
            rewards[firm_id] = float(
                self.state.revenue[firm_id]
                - 0.1 * y[firm_id]
                - fines[firm_id]
                - 0.04 * invest
                - 0.05 * lobby
                - 0.04 * evade
            )
        alive = self.state.alive.copy()
        if self.collapse and self.state.quarter == 0:
            alive[: self.firms.n // 2] = False
        if self.kill_largest and self.state.quarter == 0:
            alive[int(np.argmax(self.firms.size))] = False
        self.state = replace(
            self.state,
            y=y * alive,
            alive=alive,
            audited=audited,
            fines=fines,
            tenure=np.where(y > 0, self.state.tenure + 1, 0),
            quarter=self.state.quarter + 1,
        )
        self._lobby = next_lobby
        self.last_firm_rewards = rewards
        self.last_strategic_controls = dict(strategic_actions)
        self.last_outcome = self._outcome(int(audited.sum()))
        return None


def fake_factory(*, collapse: bool = False, kill_largest: bool = False):
    def make(cfg: RegWorldConfig, seed: int) -> FakeRegulationModel:
        return FakeRegulationModel(cfg, seed, collapse=collapse, kill_largest=kill_largest)

    return make


def test_gymnasium_checker_and_deterministic_reset(smoke_cfg: RegWorldConfig) -> None:
    env = AbmEnv(smoke_cfg, model_factory=fake_factory())
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=17)
    env.step(np.array([0.7, 0.2, 0.4, 0.1], dtype=np.float32))
    second, _ = env.reset(seed=17)
    np.testing.assert_array_equal(first, second)


def test_time_limit_is_only_truncation(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.horizon_quarters = 2
    env = AbmEnv(cfg, model_factory=fake_factory())
    env.reset(seed=1)
    assert env.step(np.zeros(4, np.float32))[2:4] == (False, False)
    assert env.step(np.zeros(4, np.float32))[2:4] == (False, True)


def test_collapse_is_only_termination(smoke_cfg: RegWorldConfig) -> None:
    env = AbmEnv(smoke_cfg, model_factory=fake_factory(collapse=True))
    env.reset(seed=1)
    assert env.step(np.zeros(4, np.float32))[2:4] == (True, False)


def test_real_regulation_model_one_controlled_step(smoke_cfg: RegWorldConfig) -> None:
    template = FakeRegulationModel(smoke_cfg, seed=3)
    n = template.firms.n
    graphs = Graphs(
        supply_und=sparse.csr_matrix((n, n)),
        influence=sparse.csr_matrix(np.eye(1)),
        market_mask=np.ones((1, n), dtype=bool),
    )
    world = ObservedWorld(
        template.firms,
        template.segments,
        graphs,
        template.state,
        Theta(beta_capacity=0.0),
    )

    def factory(cfg: RegWorldConfig, seed: int) -> RegulationModel:
        return RegulationModel(cfg, world=world, seed=seed)

    env = AbmEnv(smoke_cfg, model_factory=factory)
    env.reset(seed=11)
    observation, reward, terminated, truncated, _ = env.step(
        np.array([0.5, 0.0, 0.4, 0.2], dtype=np.float32)
    )
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not (terminated and truncated)


# --------------------------------------------------------------------------- #
# Stage 8, Phase 5 half: EmulatorEnv — identical spaces, imagination stepping #
# --------------------------------------------------------------------------- #

import torch  # noqa: E402

from regworld.environments.emulator_env import EmulatorEnv  # noqa: E402
from regworld.models.world_model import Decoded, ModelState, WorldModel  # noqa: E402
from regworld.training.datamodule import aggregate_dim  # noqa: E402

from .test_dynamics_shapes import tiny_template  # noqa: E402

_N_FIRMS, _N_SEGMENTS = 12, 3


def _emulator_meta(cfg: RegWorldConfig) -> dict:
    aggregates = torch.zeros(aggregate_dim(cfg))
    aggregates[2] = 800.0  # baseline HHI
    aggregates[3] = 0.55  # baseline trust
    aggregates[4] = 3.0  # baseline consumer surplus
    firm = torch.zeros(_N_FIRMS, 4)
    firm[:, 1] = 1.0
    firm[:, 2] = 0.35
    return {
        "initial": {
            "firm": firm,
            "segment": torch.full((_N_SEGMENTS, 1), 0.55),
            "aggregate": aggregates,
        },
        "aggregate_names": [f"a{i}" for i in range(aggregate_dim(cfg))],
        "extras": {"n_firms": _N_FIRMS},
    }


def _tiny_world_model(cfg: RegWorldConfig, seed: int = 0) -> WorldModel:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return WorldModel(
        arch="rssm_gnn",
        static_features={
            "firm": torch.randn(_N_FIRMS, 5),
            "segment": torch.randn(_N_SEGMENTS, 2),
            "association": torch.ones(2, 1),
            "regulator": torch.ones(1, 1),
        },
        aggregate_dim=aggregate_dim(cfg),
        action_dim=4,
        deter_dim=16,
        hidden_dim=32,
        latent_categories=8,
        latent_classes=8,
        gnn_layers=2,
        template=tiny_template(rng),
    )


class _ScriptedWorldModel:
    """Duck-typed stand-in emitting a fixed aggregate row every step."""

    def __init__(self, cfg: RegWorldConfig, aggregates: np.ndarray) -> None:
        self.aggregate_dim = aggregate_dim(cfg)
        self._row = torch.as_tensor(aggregates, dtype=torch.float32).unsqueeze(0)

    def eval(self) -> _ScriptedWorldModel:
        return self

    def initial_state(self, firm, segment, aggregates, generator=None) -> ModelState:
        return ModelState(
            core=torch.zeros(1, 4),
            node_hidden=None,
            firm_dynamic=firm,
            segment_dynamic=segment,
        )

    def imagine_step(self, state, action, generator=None):
        decoded = Decoded(
            aggregates=self._row.clone(),
            node_probs=torch.full((1, _N_FIRMS), 0.5),
            reward=torch.zeros(1),
            continue_prob=torch.ones(1),
        )
        return state, decoded


def test_emulator_env_checker_and_deterministic_reset(smoke_cfg: RegWorldConfig) -> None:
    env = EmulatorEnv(smoke_cfg, model=_tiny_world_model(smoke_cfg), meta=_emulator_meta(smoke_cfg))
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=17)
    env.step(np.array([0.7, 0.2, 0.4, 0.1], dtype=np.float32))
    second, _ = env.reset(seed=17)
    np.testing.assert_array_equal(first, second)
    # stochastic latents are seeded per reset: same seed, same one-step outcome
    obs_a = env.step(np.array([0.5, 0.0, 0.4, 0.2], dtype=np.float32))[0]
    env.reset(seed=17)
    obs_b = env.step(np.array([0.5, 0.0, 0.4, 0.2], dtype=np.float32))[0]
    np.testing.assert_array_equal(obs_a, obs_b)


def test_space_identity_between_abm_and_emulator(smoke_cfg: RegWorldConfig) -> None:
    """The identity that makes the planning-utility comparison possible."""
    abm = AbmEnv(smoke_cfg, model_factory=fake_factory())
    emulator = EmulatorEnv(
        smoke_cfg, model=_tiny_world_model(smoke_cfg), meta=_emulator_meta(smoke_cfg)
    )
    assert abm.observation_space == emulator.observation_space
    assert abm.action_space == emulator.action_space


def test_emulator_time_limit_is_only_truncation(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.horizon_quarters = 2
    benign = np.zeros(aggregate_dim(cfg))
    benign[0] = 0.5  # healthy compliance, no exits
    env = EmulatorEnv(cfg, model=_ScriptedWorldModel(cfg, benign), meta=_emulator_meta(cfg))
    env.reset(seed=1)
    assert env.step(np.zeros(4, np.float32))[2:4] == (False, False)
    assert env.step(np.zeros(4, np.float32))[2:4] == (False, True)


def test_emulator_collapse_is_only_termination(smoke_cfg: RegWorldConfig) -> None:
    collapsed = np.zeros(aggregate_dim(smoke_cfg))
    collapsed[5] = 0.5  # > 40% of firms exited: absorbing end, no future value
    env = EmulatorEnv(
        smoke_cfg, model=_ScriptedWorldModel(smoke_cfg, collapsed), meta=_emulator_meta(smoke_cfg)
    )
    env.reset(seed=1)
    assert env.step(np.zeros(4, np.float32))[2:4] == (True, False)


def test_emulator_reward_flag_switches_source(smoke_cfg: RegWorldConfig) -> None:
    benign = np.zeros(aggregate_dim(smoke_cfg))
    benign[0] = 0.6  # compliance up...
    benign[2], benign[3], benign[4] = 800.0, 0.55, 3.0  # ...everything else at baseline
    meta = _emulator_meta(smoke_cfg)
    env = EmulatorEnv(smoke_cfg, model=_ScriptedWorldModel(smoke_cfg, benign), meta=meta)
    env.reset(seed=1)
    recomputed = env.step(np.zeros(4, np.float32))[1]
    assert recomputed > 0.0  # compliance up from a zero baseline
    cfg_head = smoke_cfg.model_copy(deep=True)
    cfg_head.emulator.reward_from_outcomes = False
    env_head = EmulatorEnv(cfg_head, model=_ScriptedWorldModel(cfg_head, benign), meta=meta)
    env_head.reset(seed=1)
    assert env_head.step(np.zeros(4, np.float32))[1] == 0.0  # scripted head says 0
