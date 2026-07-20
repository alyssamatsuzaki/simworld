"""Stage-9 PettingZoo Parallel contract and strategic-action tests."""

from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from regworld.environments.marl_env import RegulationMARLEnv
from regworld.types import RegWorldConfig

from .test_env_contract import FakeRegulationModel, fake_factory


def _actions(env: RegulationMARLEnv) -> dict[str, np.ndarray]:
    return {agent: env.action_space(agent).sample() * 0 for agent in env.agents}


def test_parallel_api(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.env.n_strategic_firms = 3
    parallel_api_test(RegulationMARLEnv(cfg, model_factory=fake_factory()), num_cycles=100)


def test_strategic_controls_and_profit_rewards_are_real(
    smoke_cfg: RegWorldConfig,
) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.env.n_strategic_firms = 2
    env = RegulationMARLEnv(cfg, model_factory=fake_factory())
    env.reset(seed=9)
    actions = _actions(env)
    actions["regulator_0"] = np.array([0.7, 0.0, 0.5, 0.0], np.float32)
    actions["firm_0"] = np.array([0.8, 1.0, 0.6], np.float32)
    _, rewards, _, _, infos = env.step(actions)
    assert isinstance(env.model, FakeRegulationModel)
    firm_id = int(infos["firm_0"]["firm_id"])
    np.testing.assert_allclose(env.model.last_strategic_controls[firm_id], actions["firm_0"])
    assert rewards["firm_0"] == env.model.last_firm_rewards[firm_id]
    assert env.model._lobby[env.model.firms.association[firm_id]] == 1.0


def test_dead_firm_terminates_without_regulator(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.env.n_strategic_firms = 2
    env = RegulationMARLEnv(cfg, model_factory=fake_factory(kill_largest=True))
    env.reset(seed=4)
    _, _, terminated, truncated, _ = env.step(_actions(env))
    assert terminated["firm_0"] and not truncated["firm_0"]
    assert "firm_0" not in env.agents
    assert not terminated["regulator_0"]
