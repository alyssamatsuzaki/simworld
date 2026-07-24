"""Stage-10d tests: the single-agent view, iterated best response, and C6.

Everything here runs on the deterministic ``FakeRegulationModel`` backend from
``test_env_contract`` (the same one ``test_marl_env`` uses), so no Stage-1
artifacts are needed and the whole file stays inside a couple of seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from simworld.agents.marl import (
    BACKEND,
    C6_HEADLINE_METRICS,
    C6_METRICS,
    SCHEMA,
    ObservationEncoder,
    PolicyBook,
    SingleAgentView,
    compare_arms,
    comparison_path,
    constant_action_fn,
    episode_seed,
    evaluate_arm,
    reference_book,
    rule_based_firm_fn,
    train_marl,
)
from simworld.environments.marl_env import RegulationMARLEnv
from simworld.types import SimWorldConfig

from .test_env_contract import FakeRegulationModel, fake_factory


def _cfg(smoke_cfg: SimWorldConfig, *, horizon: int = 4, n_strategic: int = 3) -> SimWorldConfig:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.horizon_quarters = horizon
    cfg.env.n_strategic_firms = n_strategic
    return cfg


# --------------------------------------------------------------------------- #
# observation encoding                                                        #
# --------------------------------------------------------------------------- #


def test_encoder_flattens_dict_observations_into_unit_box(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg)
    env = RegulationMARLEnv(cfg, model_factory=fake_factory())
    observations, _ = env.reset(seed=3)

    firm_encoder = ObservationEncoder(env.observation_space("firm_0"))
    encoded = firm_encoder(observations["firm_0"])
    # 'global' (31 at the smoke profile) then 'local' (15), in sorted key order
    assert encoded.shape == (observations["firm_0"]["global"].size + 15,)
    assert firm_encoder.space.contains(encoded)
    np.testing.assert_allclose(encoded[-15:], observations["firm_0"]["local"], atol=1e-6)

    regulator_encoder = ObservationEncoder(env.observation_space("regulator_0"))
    assert regulator_encoder.space.contains(regulator_encoder(observations["regulator_0"]))


# --------------------------------------------------------------------------- #
# the single-agent view                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("role", ["regulator", "firm"])
def test_single_agent_view_passes_gymnasium_checker(smoke_cfg: SimWorldConfig, role: str) -> None:
    view = SingleAgentView(
        _cfg(smoke_cfg), role, reference_book(smoke_cfg), model_factory=fake_factory()
    )
    check_env(view, skip_render_check=True)
    view.close()


def test_view_truncates_at_horizon_and_never_terminates_there(
    smoke_cfg: SimWorldConfig,
) -> None:
    cfg = _cfg(smoke_cfg, horizon=3)
    view = SingleAgentView(cfg, "regulator", reference_book(cfg), model_factory=fake_factory())
    view.reset(seed=5)
    flags = [view.step(np.zeros(4, np.float32))[2:4] for _ in range(cfg.horizon_quarters)]
    assert flags[:-1] == [(False, False)] * (cfg.horizon_quarters - 1)
    assert flags[-1] == (False, True)
    view.close()


def test_view_terminates_when_the_ego_firm_exits(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, n_strategic=2)
    view = SingleAgentView(
        cfg, "firm", reference_book(cfg), model_factory=fake_factory(kill_largest=True)
    )
    view.reset(seed=4)
    assert view.ego == "firm_0"  # the largest strategic firm, which the fake kills
    _obs, _reward, terminated, truncated, _info = view.step(np.zeros(3, np.float32))
    assert terminated and not truncated
    view.close()


def test_ego_rotates_over_every_strategic_firm(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, n_strategic=3)
    view = SingleAgentView(cfg, "firm", reference_book(cfg), model_factory=fake_factory())
    egos = []
    for _ in range(2 * len(view.ego_pool)):
        view.reset()
        egos.append(view.ego)
    assert set(egos) == set(view.ego_pool)
    assert egos[: len(view.ego_pool)] == list(view.ego_pool)
    view.close()


def test_reset_with_the_same_seed_replays_the_same_episode(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, n_strategic=3)
    view = SingleAgentView(cfg, "firm", reference_book(cfg), model_factory=fake_factory())
    first, _ = view.reset(seed=11)
    view.step(np.ones(3, np.float32))
    second, _ = view.reset(seed=11)
    np.testing.assert_array_equal(first, second)
    view.close()


def test_frozen_opponents_actually_act_in_the_underlying_model(
    smoke_cfg: SimWorldConfig,
) -> None:
    """The non-ego firms must reach the ABM's strategic-control hook, not vanish."""
    cfg = _cfg(smoke_cfg, n_strategic=3)
    book = PolicyBook(
        regulator=constant_action_fn(np.zeros(4, np.float32)),
        firm=constant_action_fn(np.array([1.0, 1.0, 1.0], np.float32)),
    )
    view = SingleAgentView(cfg, "firm", book, model_factory=fake_factory())
    view.reset(seed=2)
    view.step(np.zeros(3, np.float32))
    model = view._env.model
    assert isinstance(model, FakeRegulationModel)
    controls = model.last_strategic_controls
    ego_id = view._env._strategic_ids[view.ego]
    np.testing.assert_allclose(controls[ego_id], np.zeros(3))
    for agent, firm_id in view._env._strategic_ids.items():
        if agent != view.ego:
            np.testing.assert_allclose(controls[firm_id], np.ones(3))
    view.close()


def test_stepping_a_finished_view_raises(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, horizon=1)
    view = SingleAgentView(cfg, "regulator", reference_book(cfg), model_factory=fake_factory())
    view.reset(seed=1)
    assert view.step(np.zeros(4, np.float32))[3] is True
    with pytest.raises(RuntimeError):
        view.step(np.zeros(4, np.float32))
    view.close()


# --------------------------------------------------------------------------- #
# C6 comparison machinery                                                     #
# --------------------------------------------------------------------------- #


def test_rule_based_arm_sends_neutral_strategic_actions(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, horizon=3, n_strategic=3)
    env = RegulationMARLEnv(cfg, model_factory=fake_factory())
    from simworld.agents.marl import rollout_arm

    rollout_arm(env, reference_book(cfg), seed=7)
    model = env.model
    assert isinstance(model, FakeRegulationModel)
    for action in model.last_strategic_controls.values():
        np.testing.assert_allclose(action, np.zeros(3))


def test_evaluate_arm_returns_every_c6_metric_per_episode(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, horizon=3, n_strategic=3)
    seeds = [episode_seed(0, index) for index in range(4)]
    arm = evaluate_arm(cfg, reference_book(cfg), seeds, model_factory=fake_factory())
    for name in C6_METRICS:
        assert name in arm and len(arm[name]) == 4
        assert all(np.isfinite(value) for value in arm[name])


def test_overlapping_cis_read_as_unchanged_and_separated_ones_as_changed() -> None:
    identical = {name: [0.5, 0.5, 0.5, 0.5] for name in C6_METRICS}
    verdict = compare_arms(identical, {name: list(values) for name, values in identical.items()})
    assert verdict["any_changed"] is False
    assert verdict["changed_metrics"] == []
    assert all(verdict["metrics"][name]["ci_overlap"] for name in C6_METRICS)

    shifted = {name: [10.0, 10.0, 10.0, 10.0] for name in C6_METRICS}
    moved = compare_arms(identical, shifted)
    assert moved["any_changed"] is True
    assert set(moved["changed_metrics"]) == set(C6_HEADLINE_METRICS)
    block = moved["metrics"]["terminal_compliance"]
    assert block["diff"] == pytest.approx(9.5)
    assert block["diff_ci95"] == pytest.approx([9.5, 9.5])


def test_compare_arms_covers_every_headline_metric() -> None:
    empty = {name: [0.0] for name in C6_METRICS}
    verdict = compare_arms(empty, empty)
    assert set(verdict["metrics"]) == set(C6_METRICS)
    assert set(C6_HEADLINE_METRICS) <= set(verdict["metrics"])


# --------------------------------------------------------------------------- #
# end to end                                                                  #
# --------------------------------------------------------------------------- #


def test_train_marl_writes_a_wireable_c6_artifact(smoke_cfg: SimWorldConfig) -> None:
    cfg = _cfg(smoke_cfg, horizon=3, n_strategic=2)
    cfg.rl.marl_timesteps = 256
    result = train_marl(cfg, model_factory=fake_factory(), n_eval_episodes=4)

    assert result.comparison == comparison_path(cfg)
    assert result.comparison.is_file() and result.summary.is_file()
    assert all(path.is_file() for path in result.checkpoints)

    payload = json.loads(result.comparison.read_text())
    assert payload["schema"] == SCHEMA
    assert payload["claim"] == "C6"
    assert payload["backend"] == BACKEND
    assert payload["degraded"] is True
    assert payload["n_eval_episodes"] == 4
    assert set(payload["arms"]) == {"rule_based", "strategic", "strategic_marl_regulator"}
    for arm in payload["arms"].values():
        assert arm["n"] == 4
        for name in C6_METRICS:
            stats = arm["metrics"][name]
            assert stats["n"] == 4
            assert stats["ci95"][0] <= stats["mean"] <= stats["ci95"][1]

    comparison = payload["comparison"]
    assert comparison["baseline_arm"] == "rule_based"
    assert comparison["strategic_arm"] == "strategic"
    assert isinstance(comparison["any_changed"], bool)
    assert comparison["any_changed"] == bool(comparison["changed_metrics"])
    for name in C6_METRICS:
        block = comparison["metrics"][name]
        assert block["changed"] is not block["ci_overlap"]

    assert payload["training"]["rounds"] >= 1
    assert len(payload["training"]["history"]) == 2 * payload["training"]["rounds"]
    assert {entry["agent"] for entry in payload["training"]["history"]} == {"firm", "regulator"}
    assert result.metrics["c6_any_changed"] == float(comparison["any_changed"])


def test_trained_firm_policy_is_not_the_rule_based_control(smoke_cfg: SimWorldConfig) -> None:
    """A learned arm that emits zeros would make C6 vacuously null."""
    cfg = _cfg(smoke_cfg, horizon=3, n_strategic=2)
    cfg.rl.marl_timesteps = 256
    train_marl(cfg, model_factory=fake_factory(), n_eval_episodes=2)

    from stable_baselines3 import PPO

    from simworld.agents.marl import marl_dir, sb3_action_fn

    agent = PPO.load(str(marl_dir(cfg) / "firm_policy.zip"), device="cpu")
    env = RegulationMARLEnv(cfg, model_factory=fake_factory())
    observations, _ = env.reset(seed=3)
    encoder = ObservationEncoder(env.observation_space("firm_0"))
    action = sb3_action_fn(agent, encoder)(observations["firm_0"])
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert not np.allclose(action, rule_based_firm_fn()(observations["firm_0"]))
