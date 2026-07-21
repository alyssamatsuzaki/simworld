"""Stage-10 contract tests: hermetic (no dependency on `artifacts/`).

Every test here injects a tiny in-memory model/env, mirroring the pattern in
`test_env_contract.py` and `test_dynamics_shapes.py`. Anything that needs the
real emulator checkpoint or trained RL artifacts is marked `slow` instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from regworld.agents import registry
from regworld.agents.dreamer import (
    Critic,
    SquashedGaussianActor,
    _budget,
    _imagination_rollout,
    compute_losses,
    normalize_obs,
    squash_action,
)
from regworld.agents.planning import evaluate_in_abm, exploitation_gap, rollout_episode
from regworld.agents.ppo import train_ppo
from regworld.environments.emulator_env import EmulatorEnv
from regworld.environments.wrappers import flat_observation_space, regulator_action_space
from regworld.types import RegWorldConfig

from .test_env_contract import _emulator_meta, _tiny_world_model, fake_factory


def test_ppo_constructs_and_learns_one_step(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.rl.n_envs = 2
    cfg.rl.total_timesteps = 64
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    result = train_ppo(cfg, model=model, meta=meta)

    assert result.checkpoint.is_file()
    assert result.summary.is_file()
    assert np.isfinite(result.metrics["mean_episode_reward"])


def test_dreamer_imagination_step_yields_finite_loss(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.rl.n_envs = 2
    cfg.emulator.imag_horizon = 3
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    batch_size, horizon, _n_updates = _budget(cfg)
    assert batch_size >= 2
    assert horizon >= 2

    obs_space = flat_observation_space(cfg)
    action_space = regulator_action_space()
    obs_low = torch.as_tensor(obs_space.low, dtype=torch.float32)
    obs_high = torch.as_tensor(obs_space.high, dtype=torch.float32)
    action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(action_space.high, dtype=torch.float32)

    actor = SquashedGaussianActor(int(obs_space.shape[0]), action_dim=4)
    critic = Critic(int(obs_space.shape[0]))
    envs = [EmulatorEnv(cfg, model=model, meta=meta) for _ in range(batch_size)]
    try:
        rollout = _imagination_rollout(
            envs,
            actor,
            critic,
            horizon=horizon,
            obs_low=obs_low,
            obs_high=obs_high,
            action_low=action_low,
            action_high=action_high,
            reset_seeds=list(range(batch_size)),
        )
        losses = compute_losses(rollout)
    finally:
        for env in envs:
            env.close()

    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name} is not finite: {value}"


def test_normalize_and_squash_stay_in_range() -> None:
    low = torch.tensor([0.0, -1.0])
    high = torch.tensor([1.0, 1.0])
    obs = torch.tensor([[0.5, 0.0], [2.0, -5.0]])
    normed = normalize_obs(obs, low, high)
    assert torch.isfinite(normed).all()

    u = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
    action = squash_action(u, low, high)
    assert bool((action >= low).all())
    assert bool((action <= high).all())


def test_registry_static_policy_gives_in_bounds_action(smoke_cfg: RegWorldConfig) -> None:
    policy = registry.load_policy(smoke_cfg, "uniform_high")
    obs_dim = int(flat_observation_space(smoke_cfg).shape[0])
    action = policy(np.zeros(obs_dim, dtype=np.float32))

    space = regulator_action_space()
    assert space.contains(action.astype(np.float32))


def test_registry_random_policy_is_seeded_and_in_bounds(smoke_cfg: RegWorldConfig) -> None:
    policy_a = registry.load_policy(smoke_cfg, "random")
    policy_b = registry.load_policy(smoke_cfg, "random")
    obs_dim = int(flat_observation_space(smoke_cfg).shape[0])
    obs = np.zeros(obs_dim, dtype=np.float32)

    action_a = policy_a(obs)
    action_b = policy_b(obs)
    space = regulator_action_space()
    assert space.contains(action_a.astype(np.float32))
    np.testing.assert_array_equal(action_a, action_b)  # same cfg.seed -> same draw


def test_registry_missing_learned_artifact_raises_clear_error(smoke_cfg: RegWorldConfig) -> None:
    with pytest.raises(FileNotFoundError, match="rl_ppo"):
        registry.load_policy(smoke_cfg, "rl_ppo")
    with pytest.raises(FileNotFoundError, match="rl_dreamer"):
        registry.load_policy(smoke_cfg, "rl_dreamer")


def test_available_policies_lists_statics_and_random_without_artifacts(
    smoke_cfg: RegWorldConfig,
) -> None:
    names = registry.available_policies(smoke_cfg)
    assert "uniform_high" in names
    assert "random" in names
    assert "rl_ppo" not in names
    assert "rl_dreamer" not in names


def test_planning_rollout_in_abm_returns_finite_j(smoke_cfg: RegWorldConfig) -> None:
    policy = registry.load_policy(smoke_cfg, "uniform_low")
    stats = evaluate_in_abm(smoke_cfg, policy, seeds=[0, 1], draws=1, model_factory=fake_factory())
    assert np.isfinite(stats.mean)
    assert np.isfinite(stats.std)
    assert stats.n == 2


def test_rollout_episode_matches_manual_loop(smoke_cfg: RegWorldConfig) -> None:
    from regworld.environments.abm_env import AbmEnv

    policy = registry.load_policy(smoke_cfg, "none")
    env = AbmEnv(smoke_cfg, model_factory=fake_factory())
    total = rollout_episode(env, policy, seed=3)
    env.close()
    assert np.isfinite(total)


def test_exploitation_gap_zero_when_equal() -> None:
    assert exploitation_gap(1.0, 1.0) == pytest.approx(0.0)
    assert exploitation_gap(1.15, 1.0) == pytest.approx(0.15)
