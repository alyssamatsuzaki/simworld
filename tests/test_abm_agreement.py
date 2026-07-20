"""Stage 3b: tensorized shapes, gradients, determinism, and Mesa agreement."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.stats import ks_2samp

from regworld.abm.tensorized import rollout_tensorized, scipy_to_torch_sparse
from regworld.rules import (
    Constants,
    FirmAttributes,
    Graphs,
    PolicyLevers,
    SegmentAttributes,
    Theta,
    initial_state,
    step_quarter,
)
from regworld.types import RegWorldConfig, validate_config

from .conftest import compose_cfg


def _config(*, n_firms: int = 48, n_segments: int = 4, horizon: int = 6) -> RegWorldConfig:
    return validate_config(
        compose_cfg(
            "profile=smoke",
            f"population.n_firms={n_firms}",
            f"population.n_consumer_segments={n_segments}",
            f"horizon_quarters={horizon}",
            "device=cpu",
        )
    )


def _world(cfg: RegWorldConfig, seed: int = 812) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    n = cfg.population.n_firms
    n_segments = cfg.population.n_consumer_segments
    size = rng.lognormal(0.0, 0.7, n)
    size /= np.median(size)
    sector = np.arange(n, dtype=np.int64) % cfg.population.n_sectors
    association = np.arange(n, dtype=np.int64) % cfg.population.n_associations
    association[::7] = -1
    size_cut = np.quantile(size, [1 / 3, 2 / 3])
    firms = FirmAttributes(
        size=size,
        sector=sector,
        data_intensity=np.clip(rng.beta(2, 2, n), 0.02, 0.98),
        cost_coef=rng.gamma(2.0, 0.5, n),
        quality=rng.normal(0.0, 1.0, n),
        base_margin=rng.beta(5, 20, n) * 0.5,
        z=np.zeros(n),
        association=association,
        region=np.zeros(n, dtype=np.int64),
        size_tercile=np.digitize(size, size_cut).astype(np.int64),
    )
    weight = np.full(n_segments, 1.0 / n_segments)
    segments = SegmentAttributes(
        weight=weight,
        privacy=np.linspace(0.2, 0.8, n_segments),
        budget=weight * size.sum(),
        trust0=np.linspace(0.55, 0.75, n_segments),
    )

    rows = np.arange(n)
    cols = (rows + 1) % n
    supply = sparse.coo_matrix(
        (np.ones(2 * n), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    ).tocsr()
    influence_dense = np.zeros((n_segments, n_segments))
    for segment in range(n_segments):
        influence_dense[segment, (segment - 1) % n_segments] = 0.5
        influence_dense[segment, (segment + 1) % n_segments] = 0.5
    market_mask = np.zeros((n_segments, n), dtype=bool)
    for firm in range(n):
        market_mask[firm % n_segments, firm] = True
        market_mask[(firm + 1) % n_segments, firm] = True
    graphs = Graphs(
        supply_und=supply,
        influence=sparse.csr_matrix(influence_dense),
        market_mask=market_mask,
    )
    constants = Constants()
    state = initial_state(firms, segments, graphs, constants, np.random.default_rng(seed + 1))
    return SimpleNamespace(
        firms=firms,
        segments=segments,
        graphs=graphs,
        initial_state=state,
        theta=Theta(),
        constants=constants,
    )


def test_tensorized_shapes_sparse_graph_and_determinism() -> None:
    cfg = _config()
    world = _world(cfg)
    policy = PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.2)
    first = rollout_tensorized(cfg, world, Theta(), policy, seed=19)
    second = rollout_tensorized(cfg, world, Theta(), policy, seed=19)
    different = rollout_tensorized(cfg, world, Theta(), policy, seed=20)

    assert first.outcome_matrix().shape == (cfg.horizon_quarters, 7)
    assert first.compliance_probabilities.shape == (
        cfg.horizon_quarters,
        cfg.population.n_firms,
    )
    assert first.final_state.spend.shape == (
        cfg.population.n_consumer_segments,
        cfg.population.n_firms,
    )
    assert len(first.covariates) == cfg.horizon_quarters
    assert torch.isfinite(first.outcome_matrix()).all()
    assert torch.equal(first.outcome_matrix(), second.outcome_matrix())
    assert torch.equal(first.final_state.y, second.final_state.y)
    assert not torch.equal(first.compliance_probabilities, different.compliance_probabilities)

    adjacency = scipy_to_torch_sparse(
        world.graphs.supply_und,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert adjacency.layout == torch.sparse_coo
    assert adjacency.is_coalesced()


def test_tensorized_zero_quarter_shapes() -> None:
    cfg = _config()
    world = _world(cfg)
    trajectory = rollout_tensorized(
        cfg,
        world,
        Theta(),
        PolicyLevers(),
        seed=0,
        quarters=0,
    )
    assert trajectory.outcome_matrix().shape == (0, 7)
    assert trajectory.compliance_probabilities.shape == (0, cfg.population.n_firms)
    assert trajectory.final_state.quarter == world.initial_state.quarter


def test_tensorized_rollout_has_finite_nonzero_gradients() -> None:
    cfg = _config()
    world = _world(cfg)
    beta_enforce = torch.tensor(2.5, requires_grad=True)
    gamma_scale = torch.tensor(0.45, requires_grad=True)
    enforcement = torch.tensor(0.6, requires_grad=True)
    theta = replace(Theta(), beta_enforce=beta_enforce, gamma_scale=gamma_scale)
    policy = PolicyLevers(
        enforcement=enforcement,
        targeting=0.5,
        phase_speed=0.3,
        subsidy=0.2,
    )
    trajectory = rollout_tensorized(cfg, world, theta, policy, seed=7, quarters=4)
    loss = trajectory.compliance_probabilities.mean() + trajectory.outcome_matrix()[:, 0].mean()
    loss.backward()

    for value in (beta_enforce, gamma_scale, enforcement):
        assert value.grad is not None
        assert torch.isfinite(value.grad)
        assert abs(float(value.grad)) > 1e-8


def test_tensorized_matches_shared_rule_distribution() -> None:
    """Fast parity check before the slower end-to-end Mesa distribution gate."""
    cfg = _config()
    policy = PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.2)
    theta = Theta()
    constants = Constants()
    numpy_terminal: list[float] = []
    torch_terminal: list[float] = []
    for seed in range(16):
        world = _world(cfg)
        state = world.initial_state
        rng = np.random.default_rng(seed)
        for _ in range(cfg.horizon_quarters):
            state, outcome, _ = step_quarter(
                state,
                world.firms,
                world.segments,
                world.graphs,
                theta,
                constants,
                policy,
                rng,
                sticky=cfg.behavior.sticky,
                attention=cfg.behavior.attention,
            )
        numpy_terminal.append(outcome.compliance_rate)
        tensorized = rollout_tensorized(cfg, world, theta, policy, seed=seed)
        torch_terminal.append(float(tensorized.outcomes[-1].compliance_rate.detach()))

    test = ks_2samp(numpy_terminal, torch_terminal)
    assert test.pvalue > 0.05, test
    assert abs(np.mean(numpy_terminal) - np.mean(torch_terminal)) < 0.12


@pytest.mark.slow
def test_tensorized_agrees_with_mesa_across_seeds() -> None:
    """PLAN Stage 3b gate: terminal aggregate distributions agree across 32 seeds."""
    from regworld.abm.model import ObservedWorld, RegulationModel

    cfg = _config(n_firms=2_000, n_segments=20, horizon=24)
    policy = PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.2)
    theta = Theta()
    mesa_terminal: list[float] = []
    torch_terminal: list[float] = []
    for seed in range(32):
        source = _world(cfg)
        world = ObservedWorld(
            firms=source.firms,
            segments=source.segments,
            graphs=source.graphs,
            initial_state=source.initial_state,
            theta=theta,
        )
        mesa = RegulationModel(cfg, world=world, theta=theta, policy=policy, seed=seed)
        mesa_trajectory = mesa.run(cfg.horizon_quarters)
        tensorized = rollout_tensorized(cfg, world, theta, policy, seed=seed)
        mesa_terminal.append(float(mesa_trajectory.outcomes[-1].compliance_rate))
        torch_terminal.append(float(tensorized.outcomes[-1].compliance_rate.detach()))

    test = ks_2samp(mesa_terminal, torch_terminal)
    assert test.pvalue > 0.05, test
