"""Stage 3 acceptance tests for the observed-world Mesa ABM."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import mesa
import numpy as np
import pytest
from polars.testing import assert_frame_equal
from scipy import sparse

from regworld.abm.agents import AssociationAgent, FirmAgent, RegulatorAgent, SegmentAgent
from regworld.abm.collect import write_tensorized_outputs, write_trajectory_outputs
from regworld.abm.model import (
    ObservedWorld,
    RegulationModel,
    load_observed_world,
    strategic_controls_from_actions,
)
from regworld.abm.policies import STATIC_POLICIES
from regworld.abm.tensorized import rollout_tensorized
from regworld.data.generate import generate_ground_truth
from regworld.rules import PolicyLevers
from regworld.types import RegWorldConfig


@pytest.fixture(scope="module")
def observed_world(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[RegWorldConfig, ObservedWorld]:
    from regworld.types import validate_config

    from .conftest import compose_cfg

    tmp = tmp_path_factory.mktemp("abm-world")
    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    cfg.paths.root = str(tmp / "artifacts")
    cfg.paths.data = str(tmp / "artifacts/data")
    cfg.paths.graphs = str(tmp / "artifacts/graphs")
    cfg.paths.reports = str(tmp / "reports")
    generate_ground_truth(cfg)
    return cfg, load_observed_world(cfg)


def test_observed_world_starts_fresh_episode(
    observed_world: tuple[RegWorldConfig, ObservedWorld],
) -> None:
    _, world = observed_world
    state = world.initial_state
    assert state.quarter == 0
    assert state.alive.all()
    assert not state.y.any()
    assert not state.audited.any()
    assert not state.tenure.any()
    assert not state.fines.any()
    assert not state.publicity.any()
    assert not state.below_floor.any()
    assert world.theta.beta_capacity == 0.0


def test_mesa_agentset_contract(observed_world: tuple[RegWorldConfig, ObservedWorld]) -> None:
    cfg, world = observed_world
    model = RegulationModel(cfg, world=world, seed=4)
    assert isinstance(model, mesa.Model)
    assert not hasattr(model, "schedule")
    assert callable(model.agents.shuffle_do)
    assert len(model.agents.select(lambda agent: isinstance(agent, FirmAgent))) == world.firms.n
    assert set(model.agents_by_type) == {
        FirmAgent,
        SegmentAgent,
        AssociationAgent,
        RegulatorAgent,
    }
    model.step()
    assert model.steps == 1
    assert model.time == 1.0


def test_deterministic_trajectory(observed_world: tuple[RegWorldConfig, ObservedWorld]) -> None:
    cfg, world = observed_world
    first = RegulationModel(cfg, world=world, seed=17).run(12)
    second = RegulationModel(cfg, world=world, seed=17).run(12)
    assert_frame_equal(first.aggregate, second.aggregate)
    assert_frame_equal(first.firm_panel, second.firm_panel)
    assert first.events == second.events
    assert np.array_equal(first.final_state.y, second.final_state.y)


def test_quarterly_invariants(observed_world: tuple[RegWorldConfig, ObservedWorld]) -> None:
    cfg, world = observed_world
    policy = STATIC_POLICIES["phased_targeted"]
    model = RegulationModel(cfg, world=world, policy=policy, seed=2)
    alive_previous = model.state.alive.copy()
    audit_limit = int(np.ceil(policy.enforcement * model.constants.audit_budget * world.firms.n))
    for _ in range(24):
        model.step()
        state = model.state
        assert np.isfinite(state.revenue).all()
        assert (state.revenue >= 0.0).all()
        assert (state.y[~state.alive] == 0.0).all()
        assert np.all(state.alive <= alive_previous)
        assert model.last_outcome is not None
        assert model.last_outcome.n_audits <= audit_limit
        assert np.isfinite(model.last_firm_rewards).all()
        assert (world.firms.base_margin >= 0.0).all()
        spend_total = state.spend.sum(axis=1)
        assert np.allclose(spend_total, world.segments.budget, atol=1e-8)
        alive_previous = state.alive.copy()


def test_enforcement_increases_terminal_compliance_on_average(
    observed_world: tuple[RegWorldConfig, ObservedWorld],
) -> None:
    cfg, world = observed_world

    def terminal(policy: PolicyLevers) -> float:
        values = [
            RegulationModel(cfg, world=world, policy=policy, seed=seed)
            .run(24)
            .outcomes[-1]
            .compliance_rate
            for seed in range(5)
        ]
        return float(np.mean(values))

    assert terminal(STATIC_POLICIES["uniform_high"]) >= terminal(STATIC_POLICIES["none"])


def test_zero_cost_and_subsidy_metamorphic_checks(
    observed_world: tuple[RegWorldConfig, ObservedWorld],
) -> None:
    cfg, world = observed_world
    zero_cost_world = replace(world, firms=replace(world.firms, cost_coef=np.zeros(world.firms.n)))
    high_enforcement = STATIC_POLICIES["uniform_high"]
    zero_cost_terminal = np.mean(
        [
            RegulationModel(cfg, world=zero_cost_world, policy=high_enforcement, seed=seed)
            .run(24)
            .outcomes[-1]
            .compliance_rate
            for seed in range(5)
        ]
    )
    assert zero_cost_terminal > 0.94

    def small_exit(subsidy: float) -> float:
        policy = PolicyLevers(0.6, 0.5, 0.3, subsidy)
        rates = []
        small = world.firms.size_tercile == 0
        for seed in range(5):
            final = RegulationModel(cfg, world=world, policy=policy, seed=seed).run(24).final_state
            rates.append(1.0 - float(final.alive[small].mean()))
        return float(np.mean(rates))

    assert small_exit(1.0) <= small_exit(0.0) + 0.02


def test_peer_coefficient_increases_supply_edge_correlation(
    observed_world: tuple[RegWorldConfig, ObservedWorld],
) -> None:
    cfg, world = observed_world
    edge_matrix = sparse.triu(world.graphs.supply_und, k=1).tocoo()

    def edge_correlation(values: np.ndarray) -> float:
        source = values[edge_matrix.row]
        target = values[edge_matrix.col]
        if source.std() == 0.0 or target.std() == 0.0:
            return 0.0
        return float(np.corrcoef(source, target)[0, 1])

    def mean_terminal_correlation(beta_peer: float) -> float:
        theta = replace(world.theta, beta_peer=beta_peer)
        correlations = [
            edge_correlation(
                RegulationModel(
                    cfg,
                    world=world,
                    theta=theta,
                    policy=STATIC_POLICIES["uniform_low"],
                    seed=seed,
                )
                .run(24)
                .final_state.y
            )
            for seed in range(5)
        ]
        return float(np.mean(correlations))

    assert mean_terminal_correlation(4.0) > mean_terminal_correlation(0.0)


def test_strategic_control_api(observed_world: tuple[RegWorldConfig, ObservedWorld]) -> None:
    cfg, world = observed_world
    firm_id = int(np.flatnonzero(world.firms.association >= 0)[0])
    association_id = int(world.firms.association[firm_id])
    controls = strategic_controls_from_actions(
        {firm_id: np.array([0.5, 0.4, 0.75])}, world.firms, world.initial_state.revenue
    )
    assert controls.utility_bonus[firm_id] == pytest.approx(1.0)
    assert controls.detection_multiplier[firm_id] == pytest.approx(0.4)
    assert controls.association_enforcement_multiplier[association_id] == pytest.approx(0.8)
    assert controls.action_cost[firm_id] > 0.0

    model = RegulationModel(cfg, world=world, seed=9)
    model.step_with_controls(controls=controls)
    assert model.steps == 1
    assert model.last_covariates["utility_bonus"][firm_id] == pytest.approx(1.0)
    assert np.isfinite(model.last_firm_rewards).all()


def test_outputs_are_parquet_and_json(
    observed_world: tuple[RegWorldConfig, ObservedWorld], tmp_path: Path
) -> None:
    cfg, world = observed_world
    output_cfg = cfg.model_copy(deep=True)
    output_cfg.paths.root = str(tmp_path / "artifacts")
    trajectory = RegulationModel(cfg, world=world, seed=6).run(3)
    paths = write_trajectory_outputs(output_cfg, trajectory, seed=6, policy_name="phased_targeted")
    assert {path.suffix for path in paths} == {".parquet", ".json"}
    summary = json.loads((tmp_path / "artifacts/abm/summary.json").read_text())
    assert summary["n_quarters"] == 3
    assert trajectory.aggregate.height == 3
    assert trajectory.firm_panel.height == world.firms.n * 3

    tensorized = rollout_tensorized(
        cfg,
        world,
        world.theta,
        STATIC_POLICIES["phased_targeted"],
        seed=6,
        quarters=3,
    )
    tensor_paths = write_tensorized_outputs(
        output_cfg, tensorized, seed=6, policy_name="phased_targeted"
    )
    assert {path.name for path in tensor_paths} == {
        "tensorized_trajectory.parquet",
        "tensorized_summary.json",
    }


def test_smoke_performance(observed_world: tuple[RegWorldConfig, ObservedWorld]) -> None:
    cfg, world = observed_world
    started = time.perf_counter()
    RegulationModel(cfg, world=world, seed=23).run(24)
    assert time.perf_counter() - started < 15.0
