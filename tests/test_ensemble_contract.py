"""Stage-11 acceptance: the scenario-cube builder and its defensive plumbing.

Hermetic tests use the tiny injected ``WorldModel`` from ``test_env_contract``
(no checkpoint, no calibrated posterior, no graphs needed). Anything that
needs the real trained emulator / posterior / observed-world artifacts is
marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import polars.testing
import pytest

from regworld.ensemble import cube as cube_mod
from regworld.ensemble import validation as validation_mod
from regworld.ensemble.cube import EnsembleResult, build_cube, resolve_policies, run_ensemble
from regworld.ensemble.validation import run_validation
from regworld.pipeline import Degraded
from regworld.types import RegWorldConfig

from .test_env_contract import _emulator_meta, _tiny_world_model


def _tiny_cfg(smoke_cfg: RegWorldConfig, **ensemble_overrides: object) -> RegWorldConfig:
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.compute.name = "local"
    cfg.ensemble.posterior_draws = 1
    cfg.ensemble.n_seeds = 1
    cfg.ensemble.batch_size = 64
    cfg.ensemble.policies = ["none", "uniform_low"]
    for key, value in ensemble_overrides.items():
        setattr(cfg.ensemble, key, value)
    return cfg


def test_build_cube_tiny_model_two_static_policies(smoke_cfg: RegWorldConfig) -> None:
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    frame, skipped = build_cube(cfg, model, meta)

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
    assert np.isfinite(frame["reward"].to_numpy()).all()
    assert frame["backfire"].dtype == pl.Boolean
    assert set(frame["draw"].to_list()) == {0}
    assert set(frame["seed_idx"].to_list()) == {0}
    # distinct seeds per policy (distinct torch.Generator draws per cell)
    assert frame["seed"].n_unique() == 2


def test_build_cube_is_deterministic(smoke_cfg: RegWorldConfig) -> None:
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    frame_a, _ = build_cube(cfg, model, meta)
    frame_b, _ = build_cube(cfg, model, meta)

    polars.testing.assert_frame_equal(
        frame_a.sort(["policy", "draw", "seed_idx"]), frame_b.sort(["policy", "draw", "seed_idx"])
    )


def test_serial_fallback_runs_without_ray(
    smoke_cfg: RegWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`compute=local` (or a below-threshold cell count) must never touch Ray."""
    cfg = _tiny_cfg(smoke_cfg)
    model = _tiny_world_model(cfg)
    meta = _emulator_meta(cfg)

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the Ray path must not run for a below-threshold serial job")

    monkeypatch.setattr(cube_mod, "_run_cells_ray", _fail_if_called)

    frame, skipped = build_cube(cfg, model, meta)
    assert frame.height == 2
    assert skipped == {}


def test_resolve_policies_skips_missing_learned_policy(
    smoke_cfg: RegWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cube_mod, "load_policy", None)
    resolved, skipped = resolve_policies(smoke_cfg, ["none", "rl_ppo"])
    assert set(resolved) == {"none"}
    assert skipped == {"rl_ppo": "unavailable (Stage 10 artifact not present)"}


def test_resolve_policies_skips_a_failing_learned_loader(
    smoke_cfg: RegWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raising_loader(_cfg: RegWorldConfig, _name: str) -> None:
        raise RuntimeError("artifact corrupt")

    monkeypatch.setattr(cube_mod, "load_policy", _raising_loader)
    resolved, skipped = resolve_policies(smoke_cfg, ["rl_dreamer"])
    assert resolved == {}
    assert skipped == {"rl_dreamer": "unavailable (Stage 10 artifact not present)"}


def test_resolve_policies_none_loader_is_also_skipped(
    smoke_cfg: RegWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cube_mod, "load_policy", lambda _cfg, _name: None)
    resolved, skipped = resolve_policies(smoke_cfg, ["rl_ppo"])
    assert resolved == {}
    assert "rl_ppo" in skipped


def test_run_ensemble_raises_degraded_without_checkpoint(smoke_cfg: RegWorldConfig) -> None:
    """No `artifacts/emulator/.../model.pt` under the tmp root -> an honest partial."""
    with pytest.raises(Degraded):
        run_ensemble(smoke_cfg)


def test_run_validation_reports_missing_posterior_honestly(smoke_cfg: RegWorldConfig) -> None:
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


def test_run_validation_no_static_policy_cells_is_honest(smoke_cfg: RegWorldConfig) -> None:
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


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("artifacts/emulator/rssm_gnn/model.pt").exists(),
    reason="Requires a real trained emulator checkpoint (run `make emulator` first); "
    "artifacts/ is gitignored, so a fresh checkout never has one",
)
def test_run_ensemble_end_to_end_on_real_artifacts() -> None:
    """Full run against the real trained checkpoint + calibrated posterior."""
    from regworld.types import validate_config

    from .conftest import compose_cfg

    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    result = run_ensemble(cfg)
    assert isinstance(result, EnsembleResult)
    assert result.cube.exists()
    assert result.summary.exists()
    assert result.metrics["n_cells"] > 0
