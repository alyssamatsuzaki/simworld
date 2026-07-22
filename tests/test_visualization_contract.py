"""Stage 15 acceptance: figures degrade gracefully, and the OOD banner works.

Hermetic tests write tiny synthetic artifacts under a ``smoke_cfg`` rooted in
``tmp_path`` and never touch a real checkpoint or posterior. Anything that
needs the real trained emulator, calibrated posterior, or scenario cube is
marked ``slow`` and skipped unless those artifacts already exist under the
repo's (gitignored) ``artifacts/`` root from a prior local run — e.g. after
``make emulator`` / ``make smoke`` (mirrors ``tests/test_ensemble_contract.py``'s
end-to-end pattern). They never run on a fresh checkout, including in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regworld.types import RegWorldConfig
from regworld.visualization import figures as figures_mod
from regworld.visualization import interactive as interactive_mod
from regworld.visualization.dashboard import (
    OOD_THRESHOLD,
    _fallback_train_actions,
    _match_grid_policy,
    ood_mahalanobis,
)
from regworld.visualization.figures import FIGURE_FUNCS, make_all_figures

from .conftest import compose_cfg


# --------------------------------------------------------------------- OOD --
def test_ood_mahalanobis_in_distribution_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    train_actions = rng.normal(loc=[0.5, 0.0, 0.5, 0.3], scale=0.05, size=(500, 4))
    in_dist = np.array([0.5, 0.0, 0.5, 0.3])
    assert ood_mahalanobis(in_dist, train_actions) < 0.5


def test_ood_mahalanobis_far_outside_is_large() -> None:
    rng = np.random.default_rng(0)
    train_actions = rng.normal(loc=[0.5, 0.0, 0.5, 0.3], scale=0.05, size=(500, 4))
    far = np.array([5.0, 3.0, -4.0, 6.0])
    assert ood_mahalanobis(far, train_actions) > 10 * ood_mahalanobis(
        np.array([0.5, 0.0, 0.5, 0.3]), train_actions
    )


def test_ood_threshold_separates_the_two_cases() -> None:
    rng = np.random.default_rng(1)
    train_actions = rng.normal(loc=[0.5, 0.0, 0.5, 0.3], scale=0.05, size=(500, 4))
    assert ood_mahalanobis(np.array([0.5, 0.0, 0.5, 0.3]), train_actions) < OOD_THRESHOLD
    assert ood_mahalanobis(np.array([5.0, 3.0, -4.0, 6.0]), train_actions) > OOD_THRESHOLD


def test_fallback_train_actions_stays_inside_the_action_box(smoke_cfg: RegWorldConfig) -> None:
    from regworld.visualization._io import action_bounds

    low, high = action_bounds()
    draws = _fallback_train_actions(smoke_cfg, n_samples=64)
    assert draws.shape == (64, 4)
    assert np.all(draws >= low) and np.all(draws <= high)


def test_match_grid_policy_finds_exact_and_rejects_far() -> None:
    from regworld.abm.policies import STATIC_POLICIES

    exact = STATIC_POLICIES["targeted"].as_array()
    assert _match_grid_policy(exact) == "targeted"
    assert _match_grid_policy(np.array([0.11, 0.22, 0.33, 0.44])) is None


# ------------------------------------------------------------- fig helpers --
def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_fig_four_numbers_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_json(
        Path(smoke_cfg.paths.root) / "causal" / "four_numbers.json",
        {
            "tau_true": 0.41,
            "tau_abm": 0.35,
            "tau_abm_mc_se": 0.01,
            "tau_qe": 0.06,
            "tau_qe_ci": [-0.1, 0.2],
            "tau_obs": 0.12,
            "tau_obs_ci": [0.03, 0.22],
            "flagged": False,
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_four_numbers(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_fig_four_numbers_skips_when_missing(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    assert figures_mod.fig_four_numbers(smoke_cfg, fig_dir) is None


def test_fig_event_study_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_json(
        Path(smoke_cfg.paths.root) / "causal" / "causal_estimates.json",
        {
            "did": {
                "att": 0.06,
                "se": 0.1,
                "event_times": [-2, -1, 0, 1, 2],
                "event_coefficients": [0.01, -0.02, 0.05, 0.1, 0.15],
                "pretrend_max_abs": 0.02,
            }
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_event_study(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_fig_sensitivity_tornado_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_json(
        Path(smoke_cfg.paths.root) / "sensitivity" / "indices.json",
        {
            "sobol": {
                "S1": {"enforcement": 0.1, "targeting": 0.2, "phase_speed": 0.05, "subsidy": 0.3},
                "ST": {"enforcement": 0.4, "targeting": 0.5, "phase_speed": 0.2, "subsidy": 0.6},
            }
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_sensitivity_tornado(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_fig_calibration_curve_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_json(
        Path(smoke_cfg.paths.reports) / "eval" / "metrics.json",
        {
            "calibration": {
                "ece": 0.03,
                "coverage_50": 0.48,
                "coverage_80": 0.79,
                "coverage_90": 0.88,
                "coverage_95": 0.94,
                "reliability_diagram": [
                    {"bin": "[0.0,0.1)", "confidence": 0.05, "accuracy": 0.06, "count": 10},
                    {"bin": "[0.1,0.2)", "confidence": 0.15, "accuracy": 0.12, "count": 10},
                ],
            }
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_calibration_curve(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def _write_synthetic_cube(cfg: RegWorldConfig) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for policy, base_compliance, base_hhi, backfire in (
        ("none", 0.5, 800.0, False),
        ("uniform_low", 0.6, 850.0, False),
        ("targeted", 0.7, 950.0, True),
    ):
        for i in range(6):
            rows.append(
                {
                    "compliance_rate": float(base_compliance + rng.normal(0, 0.02)),
                    "hhi": float(base_hhi + rng.normal(0, 5.0)),
                    "reward": float(10.0 + rng.normal(0, 0.5)),
                    "backfire": backfire,
                    "policy": policy,
                    "draw": i,
                    "seed_idx": 0,
                }
            )
    frame = pl.DataFrame(rows)
    out = Path(cfg.paths.root) / "ensemble" / "cube.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    return out


def test_fig_pareto_frontier_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_synthetic_cube(smoke_cfg)
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_pareto_frontier(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_fig_policy_comparison_j_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_synthetic_cube(smoke_cfg)
    _write_json(
        Path(smoke_cfg.paths.reports) / "eval" / "metrics.json",
        {
            "planning_utility": {
                "policies": {
                    "none": {"mean_return": 9.0},
                    "uniform_low": {"mean_return": 11.0},
                    "targeted": {"mean_return": 12.5},
                }
            }
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_policy_comparison_j(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_fig_ood_degradation_hermetic(smoke_cfg: RegWorldConfig, tmp_path: Path) -> None:
    _write_json(
        Path(smoke_cfg.paths.reports) / "eval" / "metrics.json",
        {
            "ood": {
                "heldout_mean_error": 0.05,
                "heldout_distances": [0.5, 0.6, 0.7],
                "enforcement_1p5_error": 0.3,
                "enforcement_1p5_mahalanobis": 4.0,
                "heldout_error_vs_mahalanobis_spearman": 0.4,
            }
        },
    )
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    out = figures_mod.fig_ood_degradation(smoke_cfg, fig_dir)
    assert out is not None and out.is_file()


def test_interactive_trajectory_fans_hermetic(smoke_cfg: RegWorldConfig) -> None:
    _write_synthetic_cube(smoke_cfg)
    fig = interactive_mod.trajectory_fans_figure(smoke_cfg)
    assert fig is not None


def test_interactive_trajectory_fans_skips_when_missing(smoke_cfg: RegWorldConfig) -> None:
    assert interactive_mod.trajectory_fans_figure(smoke_cfg) is None


# ------------------------------------------------------------ whole-of-run --
def test_figure_registry_has_thirteen_entries() -> None:
    assert len(FIGURE_FUNCS) == 13
    assert len({f.__name__ for f in FIGURE_FUNCS}) == 13


def test_make_all_figures_degrades_to_empty_without_any_artifacts(
    smoke_cfg: RegWorldConfig,
) -> None:
    """No artifacts at all under the tmp root -> an honest empty list, never a crash."""
    assert make_all_figures(smoke_cfg) == []


def test_make_all_figures_partial_artifacts_writes_only_what_it_can(
    smoke_cfg: RegWorldConfig,
) -> None:
    _write_json(
        Path(smoke_cfg.paths.root) / "causal" / "four_numbers.json",
        {"tau_true": 0.4, "tau_abm": 0.3, "tau_qe": 0.1, "tau_obs": 0.1, "flagged": False},
    )
    _write_synthetic_cube(smoke_cfg)
    written = make_all_figures(smoke_cfg)
    assert written, "at least the four-number table and the Pareto frontier should render"
    assert all(p.is_file() for p in written)
    assert len(written) < 13


# --------------------------------------------------------------------- slow -
@pytest.mark.slow
@pytest.mark.skipif(
    not Path("artifacts/emulator/rssm_gnn/model.pt").exists(),
    reason="Requires a real trained emulator checkpoint (run `make emulator` first); "
    "artifacts/ is gitignored, so a fresh checkout never has one",
)
def test_make_all_figures_end_to_end_on_real_artifacts() -> None:
    """Full run against the committed checkpoint, posterior, cube, and eval report."""
    from regworld.types import validate_config

    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    written = make_all_figures(cfg)
    assert len(written) > 0
    assert all(p.is_file() for p in written)


@pytest.mark.slow
def test_load_default_config_and_ood_banner_on_real_artifacts() -> None:
    from regworld.visualization.dashboard import _train_action_distribution, load_default_config

    cfg = load_default_config("smoke")
    train_actions = _train_action_distribution(cfg)
    assert train_actions.ndim == 2 and train_actions.shape[1] == 4
    in_dist = train_actions.mean(axis=0)
    assert ood_mahalanobis(in_dist, train_actions) < OOD_THRESHOLD
    assert ood_mahalanobis(np.array([50.0, 30.0, -40.0, 60.0]), train_actions) > OOD_THRESHOLD
