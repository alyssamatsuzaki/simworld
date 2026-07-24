"""Stage-11 acceptance: the scenario-cube builder and its defensive plumbing.

Hermetic tests use the tiny injected ``WorldModel`` from ``test_env_contract``
(no checkpoint, no calibrated posterior, no graphs needed). Anything that
needs the real trained emulator / posterior / observed-world artifacts is
marked ``slow``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import polars.testing
import pytest

from simworld import rules
from simworld.ensemble import cube as cube_mod
from simworld.ensemble import validation as validation_mod
from simworld.ensemble.cube import EnsembleResult, build_cube, resolve_policies, run_ensemble
from simworld.ensemble.validation import (
    COVERAGE_THRESHOLD_DEV,
    GATE_FAIL,
    GATE_INDETERMINATE,
    GATE_PASS,
    GATE_UNGATED,
    CoverageGateFailure,
    coverage_gate_status,
    enforce_coverage_gate,
    run_validation,
)
from simworld.pipeline import Degraded
from simworld.types import SimWorldConfig

from .test_env_contract import _emulator_meta, _tiny_world_model


def _tiny_cfg(smoke_cfg: SimWorldConfig, **ensemble_overrides: object) -> SimWorldConfig:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.compute.name = "local"
    cfg.ensemble.posterior_draws = 1
    cfg.ensemble.n_seeds = 1
    cfg.ensemble.batch_size = 64
    cfg.ensemble.policies = ["none", "uniform_low"]
    for key, value in ensemble_overrides.items():
        setattr(cfg.ensemble, key, value)
    return cfg


def test_build_cube_tiny_model_two_static_policies(smoke_cfg: SimWorldConfig) -> None:
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    frame, skipped, dataset = build_cube(cfg, model, meta)

    assert skipped == {}
    # 2 policies x 1 posterior draw x 1 seed = 2 cells.
    assert frame.height == 2
    assert sorted(frame["policy"].unique().to_list()) == ["none", "uniform_low"]
    for column in (
        "compliance_rate",
        "hhi",
        "mean_trust",
        "consumer_surplus",
        "exit_rate",
        "reward",
    ):
        assert column in frame.columns
    assert "_traj" not in frame.columns  # trajectories live in the Zarr cube, not the frame
    assert np.isfinite(frame["reward"].to_numpy()).all()
    assert frame["backfire"].dtype == pl.Boolean
    assert set(frame["draw"].to_list()) == {0}
    assert set(frame["seed_idx"].to_list()) == {0}
    # distinct seeds per policy (distinct torch.Generator draws per cell)
    assert frame["seed"].n_unique() == 2

    # §18: the cube carries the (policy, draw, seed, quarter, variable) contract.
    assert dataset["outcomes"].dims == ("policy", "draw", "seed", "quarter", "variable")
    assert list(dataset.sizes.values()) == [2, 1, 1, cfg.horizon_quarters, 9]
    assert dataset.coords["variable"].values[0] == "compliance_rate"
    assert dataset.coords["variable"].values[-1] == "backfire"
    assert list(dataset.coords["policy"].values) == ["none", "uniform_low"]
    # every run-to-horizon cell has a finite compliance trajectory at quarter 1
    q1 = dataset["outcomes"].isel(quarter=0, variable=0).values
    assert np.isfinite(q1).all()


def test_build_cube_is_deterministic(smoke_cfg: SimWorldConfig) -> None:
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    frame_a, _, _ = build_cube(cfg, model, meta)
    frame_b, _, _ = build_cube(cfg, model, meta)

    polars.testing.assert_frame_equal(
        frame_a.sort(["policy", "draw", "seed_idx"]), frame_b.sort(["policy", "draw", "seed_idx"])
    )


def test_serial_fallback_runs_without_ray(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`compute=local` (or a below-threshold cell count) must never touch Ray."""
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the Ray path must not run for a below-threshold serial job")

    monkeypatch.setattr(cube_mod, "_run_cells_ray", _fail_if_called)

    frame, skipped, _ = build_cube(cfg, model, meta)
    assert frame.height == 2
    assert skipped == {}


def test_resolve_policies_skips_missing_learned_policy(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cube_mod, "load_policy", None)
    resolved, skipped = resolve_policies(smoke_cfg, ["none", "rl_ppo"])
    assert set(resolved) == {"none"}
    assert skipped == {"rl_ppo": "unavailable (Stage 10 artifact not present)"}


def test_resolve_policies_skips_a_failing_learned_loader(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raising_loader(_cfg: SimWorldConfig, _name: str) -> None:
        raise RuntimeError("artifact corrupt")

    monkeypatch.setattr(cube_mod, "load_policy", _raising_loader)
    resolved, skipped = resolve_policies(smoke_cfg, ["rl_dreamer"])
    assert resolved == {}
    assert skipped == {"rl_dreamer": "unavailable (Stage 10 artifact not present)"}


def test_resolve_policies_none_loader_is_also_skipped(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cube_mod, "load_policy", lambda _cfg, _name: None)
    resolved, skipped = resolve_policies(smoke_cfg, ["rl_ppo"])
    assert resolved == {}
    assert "rl_ppo" in skipped


def test_run_ensemble_raises_degraded_without_checkpoint(smoke_cfg: SimWorldConfig) -> None:
    """No `artifacts/emulator/.../model.pt` under the tmp root -> an honest partial."""
    with pytest.raises(Degraded):
        run_ensemble(smoke_cfg)


def test_run_validation_reports_missing_posterior_honestly(smoke_cfg: SimWorldConfig) -> None:
    """No `artifacts/calibration/posterior.nc` under the tmp root -> NaN coverage, not a crash."""
    cube = pl.DataFrame(
        {
            "policy": ["none", "none"],
            "draw": [0, 1],
            "seed_idx": [0, 0],
            "compliance_rate": [0.1, 0.2],
            "hhi": [1000.0, 1100.0],
            "mean_trust": [0.5, 0.6],
            "consumer_surplus": [1.0, 1.2],
            "exit_rate": [0.0, 0.01],
        }
    )
    report = run_validation(smoke_cfg, cube, model=None, meta=None)  # type: ignore[arg-type]
    assert report.n_validated == 0
    assert np.isnan(report.coverage)
    assert report.path.exists()


def test_run_validation_no_static_policy_cells_is_honest(smoke_cfg: SimWorldConfig) -> None:
    cube = pl.DataFrame({"policy": [], "draw": [], "compliance_rate": []})
    report = run_validation(smoke_cfg, cube, model=None, meta=None)  # type: ignore[arg-type]
    assert report.n_validated == 0
    assert np.isnan(report.coverage)


def test_metrics_terminology_matches_validation_module() -> None:
    assert set(validation_mod.METRICS) == {
        "compliance_rate",
        "hhi",
        "mean_trust",
        "consumer_surplus",
        "exit_rate",
    }


# --------------------------------------------------------------------------- #
# Stage-11 coverage gate (§10 Stage 11, §18: coverage >= 0.85)
# --------------------------------------------------------------------------- #

_COVERING = rules.QuarterOutcome(
    compliance_rate=0.335,
    compliance_rate_weighted=0.335,
    compliance_by_tercile=(0.3, 0.33, 0.36),
    hhi=1035.0,
    mean_trust=0.535,
    consumer_surplus=1.35,
    exit_rate_cum=0.035,
    enforcement_cost=1.0,
    n_audits=3,
)
_MISSING = rules.QuarterOutcome(
    compliance_rate=0.99,
    compliance_rate_weighted=0.99,
    compliance_by_tercile=(0.99, 0.99, 0.99),
    hhi=9_000.0,
    mean_trust=0.01,
    consumer_surplus=99.0,
    exit_rate_cum=0.99,
    enforcement_cost=1.0,
    n_audits=3,
)


def _fake_cube(policies: tuple[str, ...] = ("none",), n_draws: int = 8) -> pl.DataFrame:
    """A cube whose per-policy 5%-95% bands are known by construction."""
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for draw in range(n_draws):
            rows.append(
                {
                    "policy": policy,
                    "draw": draw,
                    "seed_idx": 0,
                    "compliance_rate": 0.30 + 0.01 * draw,
                    "hhi": 1000.0 + 10.0 * draw,
                    "mean_trust": 0.50 + 0.01 * draw,
                    "consumer_surplus": 1.0 + 0.1 * draw,
                    "exit_rate": 0.0 + 0.01 * draw,
                }
            )
    return pl.DataFrame(rows)


def _stub_abm(
    monkeypatch: pytest.MonkeyPatch,
    outcome: rules.QuarterOutcome,
    *,
    n_theta: int = 512,
) -> list[tuple[str, int, float]]:
    """Replace the tensorized-ABM leg; record (policy, cube draw, theta row) per call.

    Column 0 of the fake posterior is the row index, so ``theta.beta_0`` reads
    back exactly which posterior row the validator picked.
    """
    n_fields = len(rules.Theta.__dataclass_fields__)
    theta_rows = np.zeros((n_theta, n_fields), dtype=float)
    theta_rows[:, 0] = np.arange(n_theta, dtype=float)
    calls: list[tuple[str, int, float]] = []

    def _world(_cfg: object, seed: int | None = None) -> object:
        del seed
        return object()

    monkeypatch.setattr(validation_mod, "load_theta_draws", lambda _cfg: theta_rows)
    monkeypatch.setattr(validation_mod, "load_observed_world", _world)

    def _rollout(
        _cfg: object,
        _world: object,
        theta: rules.Theta,
        policy: object,
        seed: int,
        quarters: int | None = None,
    ) -> object:
        del policy, quarters
        calls.append(("", seed, float(theta.beta_0)))
        return object()

    monkeypatch.setattr(validation_mod, "rollout_tensorized", _rollout)
    monkeypatch.setattr(validation_mod, "_terminal_tensor_outcome", lambda _traj: outcome)
    return calls


def _gating_cfg(smoke_cfg: SimWorldConfig) -> SimWorldConfig:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.profile_name = "dev"
    cfg.ensemble.validation_frac = 1.0
    return cfg


def test_coverage_gate_status_classifies_against_the_plan_threshold(
    smoke_cfg: SimWorldConfig,
) -> None:
    dev = _gating_cfg(smoke_cfg)
    assert COVERAGE_THRESHOLD_DEV == 0.85
    assert coverage_gate_status(dev, COVERAGE_THRESHOLD_DEV) == GATE_PASS
    assert coverage_gate_status(dev, 0.9) == GATE_PASS
    assert coverage_gate_status(dev, 0.8499) == GATE_FAIL
    assert coverage_gate_status(dev, 0.10) == GATE_FAIL
    assert coverage_gate_status(dev, float("nan")) == GATE_INDETERMINATE
    # smoke reports without gating, whatever the number is
    assert coverage_gate_status(smoke_cfg, 0.10) == GATE_UNGATED
    assert coverage_gate_status(smoke_cfg, float("nan")) == GATE_UNGATED


def test_enforce_coverage_gate_raises_below_threshold(smoke_cfg: SimWorldConfig) -> None:
    dev = _gating_cfg(smoke_cfg)
    assert enforce_coverage_gate(dev, 0.9) == GATE_PASS
    assert enforce_coverage_gate(smoke_cfg, 0.10) == GATE_UNGATED  # never raises at smoke
    with pytest.raises(CoverageGateFailure) as excinfo:
        enforce_coverage_gate(dev, 0.10)
    assert excinfo.value.coverage == pytest.approx(0.10)
    assert excinfo.value.threshold == COVERAGE_THRESHOLD_DEV
    assert excinfo.value.status == GATE_FAIL
    with pytest.raises(CoverageGateFailure) as nan_info:
        enforce_coverage_gate(dev, float("nan"))
    assert nan_info.value.status == GATE_INDETERMINATE


def test_bad_coverage_is_a_hard_failure_at_a_gating_profile(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of defect (1): 0.0 coverage at dev must not sail through."""
    cfg = _gating_cfg(smoke_cfg)
    _stub_abm(monkeypatch, _MISSING)

    report = run_validation(cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]

    assert report.coverage == pytest.approx(0.0)
    assert report.n_validated == 8 * len(validation_mod.METRICS)
    assert report.status == GATE_FAIL
    assert report.gated is True
    assert report.passed is False
    written = json.loads(report.path.read_text())
    assert written["status"] == GATE_FAIL
    assert written["gated"] is True
    assert written["threshold"] == COVERAGE_THRESHOLD_DEV
    with pytest.raises(CoverageGateFailure):
        report.raise_for_gate()
    with pytest.raises(CoverageGateFailure):
        enforce_coverage_gate(cfg, report.coverage, report_path=report.path)


def test_good_coverage_passes_the_gate_at_a_gating_profile(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _gating_cfg(smoke_cfg)
    _stub_abm(monkeypatch, _COVERING)

    report = run_validation(cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]

    assert report.coverage == pytest.approx(1.0)
    assert report.status == GATE_PASS
    assert report.passed is True
    report.raise_for_gate()  # must not raise


def test_smoke_reports_bad_coverage_without_gating(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.ensemble.validation_frac = 1.0
    _stub_abm(monkeypatch, _MISSING)

    report = run_validation(cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]

    assert report.coverage == pytest.approx(0.0)
    assert report.status == GATE_UNGATED
    assert report.gated is False
    assert report.passed is True
    report.raise_for_gate()  # smoke never blocks
    assert json.loads(report.path.read_text())["gated"] is False


def test_missing_posterior_at_a_gating_profile_is_indeterminate_not_pass(
    smoke_cfg: SimWorldConfig,
) -> None:
    """Graceful degradation stays graceful, but must not silently count as a pass."""
    cfg = _gating_cfg(smoke_cfg)
    report = run_validation(cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]

    assert np.isnan(report.coverage)
    assert report.n_validated == 0
    assert report.path.exists()
    assert report.status == GATE_INDETERMINATE
    assert report.passed is False


def test_smoke_degraded_paths_stay_ungated_and_write_a_report(smoke_cfg: SimWorldConfig) -> None:
    empty = run_validation(
        smoke_cfg,
        pl.DataFrame({"policy": [], "draw": [], "compliance_rate": []}),
        model=None,  # type: ignore[arg-type]
        meta=None,  # type: ignore[arg-type]
    )
    assert empty.status == GATE_UNGATED
    assert empty.passed is True
    assert empty.path.exists()

    no_posterior = run_validation(smoke_cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]
    assert no_posterior.status == GATE_UNGATED
    assert no_posterior.passed is True


def test_theta_draws_come_from_the_whole_posterior_not_the_cube_draw_index(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect (2): the cube's `draw` column is a seed index, not a posterior id.

    The old `theta_rows[draw_idx % n]` pairing could only ever reach the first
    `n_draws` rows of the posterior. Theta must be sampled across the posterior,
    and the same cube draw must map to the same theta for every policy so the
    cross-policy comparison keeps common random numbers.
    """
    cfg = _gating_cfg(smoke_cfg)
    calls = _stub_abm(monkeypatch, _COVERING, n_theta=512)

    run_validation(cfg, _fake_cube(policies=("none", "uniform_low")), model=None, meta=None)  # type: ignore[arg-type]

    assert len(calls) == 16  # 2 policies x 8 draws
    theta_used = [row for _, _, row in calls]
    assert max(theta_used) >= 8, "theta rows never reach beyond the cube's draw indices"
    # seed encodes the cube draw; the same draw must reuse the same theta row
    by_seed: dict[int, set[float]] = {}
    for _, seed, row in calls:
        by_seed.setdefault(seed, set()).add(row)
    assert len(by_seed) == 8
    assert all(len(rows) == 1 for rows in by_seed.values())


def test_validation_report_does_not_claim_a_per_draw_predictive_interval(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _gating_cfg(smoke_cfg)
    _stub_abm(monkeypatch, _COVERING)

    report = run_validation(cfg, _fake_cube(), model=None, meta=None)  # type: ignore[arg-type]
    written = json.loads(report.path.read_text())

    assert written["metric"] == "marginal_interval_coverage"
    assert "theta-marginal" in written["interval_kind"]
    assert report.interval_kind == validation_mod.INTERVAL_KIND
    policy_row = written["per_policy"][0]
    assert "emulator_marginal_interval_05_95" in policy_row
    assert "intervals" not in policy_row  # the old, misleadingly-named key
    assert set(policy_row["per_metric_coverage"]) == set(validation_mod.METRICS)


@pytest.mark.slow
def test_run_ensemble_end_to_end_on_real_artifacts() -> None:
    """Full run against the real trained checkpoint + calibrated posterior."""
    from simworld.types import validate_config

    from .conftest import compose_cfg

    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    result = run_ensemble(cfg)
    assert isinstance(result, EnsembleResult)
    assert result.cube.exists()
    assert result.summary.exists()
    assert result.metrics["n_cells"] > 0

    # §18: the Zarr cube exists with the (policy, draw, seed, quarter, variable)
    # contract, and P(backfire | policy) is recorded for every included policy.
    import json as _json

    import xarray as xr

    summary = _json.loads(result.summary.read_text())
    zarr_cube = xr.open_zarr(summary["cube_zarr_path"])
    assert zarr_cube["outcomes"].dims == ("policy", "draw", "seed", "quarter", "variable")
    assert set(summary["p_backfire_by_policy"]) == set(summary["policies_included"])
    assert all(0.0 <= v <= 1.0 for v in summary["p_backfire_by_policy"].values())
