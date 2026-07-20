"""Scientific guardrails for Regime-F initialization and DGP variants."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from regworld.data.generate import _baseline_outcome, _copy_state
from regworld.dgp import dynamics
from regworld.dgp.history import draw_rollout
from regworld.rules import (
    Constants,
    FirmAttributes,
    Graphs,
    PolicyLevers,
    QuarterOutcome,
    SegmentAttributes,
    WorldState,
    regulator_reward,
)
from regworld.types import RegWorldConfig, validate_config

from .conftest import compose_cfg


def _world() -> tuple[FirmAttributes, SegmentAttributes, Graphs, WorldState]:
    firms = FirmAttributes(
        size=np.array([1.0, 2.0]),
        sector=np.array([0, 0]),
        data_intensity=np.array([0.4, 0.6]),
        cost_coef=np.array([1.0, 1.0]),
        quality=np.array([0.0, 1.0]),
        base_margin=np.array([0.1, 0.1]),
        z=np.array([-1.0, 1.0]),
        association=np.array([0, 0]),
        region=np.array([0, 0]),
        size_tercile=np.array([0, 2]),
    )
    segments = SegmentAttributes(
        weight=np.array([1.0]),
        privacy=np.array([0.5]),
        budget=np.array([3.0]),
        trust0=np.array([0.6]),
    )
    graphs = Graphs(
        supply_und=sparse.csr_matrix((2, 2)),
        influence=sparse.csr_matrix((1, 1)),
        market_mask=np.ones((1, 2), dtype=bool),
    )
    state = WorldState(
        y=np.zeros(2),
        alive=np.ones(2, dtype=bool),
        revenue=np.array([1.0, 2.0]),
        tenure=np.zeros(2),
        fines=np.zeros(2),
        audited=np.zeros(2, dtype=bool),
        spend=np.array([[1.0, 2.0]]),
        trust=np.array([0.6]),
        publicity=np.zeros(1),
        rev_hist=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        below_floor=np.zeros(2),
        quarter=24,
    )
    return firms, segments, graphs, state


def test_future_copy_resets_policy_clock_and_uses_a_common_q0_baseline() -> None:
    firms, segments, graphs, historical_state = _world()

    future_state = _copy_state(historical_state)
    baseline = _baseline_outcome(future_state, firms, segments, graphs)

    assert historical_state.quarter == 24
    assert future_state.quarter == 0
    assert baseline.compliance_rate == 0.0
    assert baseline.enforcement_cost == 0.0
    assert np.isfinite(baseline.consumer_surplus)


def test_wellspecified_world_omits_only_the_latent_capacity_term(monkeypatch: Any) -> None:
    firms, segments, graphs, state = _world()
    baseline = _baseline_outcome(state, firms, segments, graphs)
    observed_flags: list[bool] = []

    def fake_step(
        *args: Any, **kwargs: Any
    ) -> tuple[WorldState, QuarterOutcome, dict[str, np.ndarray]]:
        observed_flags.append(bool(kwargs["use_z"]))
        return args[0], baseline, {}

    monkeypatch.setattr(dynamics, "step_quarter", fake_step)
    for variant in ("wellspecified", "confounded"):
        cfg: RegWorldConfig = validate_config(compose_cfg(f"dgp={variant}", "profile=smoke"))
        dynamics.run_dgp(
            cfg,
            firms,
            segments,
            graphs,
            PolicyLevers(),
            seed=0,
            quarters=1,
            start_state=state,
        )

    assert observed_flags == [False, True]


def test_staggered_rollout_retains_not_yet_treated_controls(
    smoke_cfg: RegWorldConfig,
) -> None:
    rollout = draw_rollout(smoke_cfg, np.random.default_rng(smoke_cfg.seed + 90_001))
    assert rollout.min() >= 2
    assert rollout.max() >= smoke_cfg.observed_quarters
    assert np.unique(rollout).size == smoke_cfg.population.n_regions


def test_reward_penalizes_future_exits_not_historical_baseline_exits() -> None:
    baseline = QuarterOutcome(0.0, 0.0, (0.0, 0.0, 0.0), 100.0, 0.5, 1.0, 0.4, 0.0, 0)
    unchanged = QuarterOutcome(0.0, 0.0, (0.0, 0.0, 0.0), 100.0, 0.5, 1.0, 0.4, 0.0, 0)

    reward = regulator_reward(
        unchanged,
        baseline,
        weights=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        const=Constants(),
        n_firms=2,
    )

    assert reward == 0.0
