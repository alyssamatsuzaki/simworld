"""Stage 4 scientific contracts: likelihood semantics and tiny recovery gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regworld.abm.tensorized import rollout_tensorized
from regworld.calibration.diagnostics import run_micro_diagnostics
from regworld.calibration.macro_smc import MACRO_PARAMETER_NAMES, sample_macro_prior
from regworld.calibration.micro_numpyro import (
    MICRO_PARAMETER_NAMES,
    MicroData,
    fit_micro_numpyro,
    fit_tiny_numpyro,
    micro_data_from_frame,
    reported_probability,
)
from regworld.calibration.summaries import SUMMARY_NAMES, summary_statistics
from regworld.rules import PolicyLevers, Theta
from regworld.stages import stage_calibration
from regworld.tracking import NullTracker
from regworld.types import RegWorldConfig

from .test_abm_agreement import _config, _world


def test_reporting_misclassification_is_marginalized() -> None:
    probability = reported_probability(np.asarray([0.0, 0.25, 1.0]), q0=0.05, q1=0.10)
    np.testing.assert_allclose(probability, [0.05, 0.2625, 0.90])


def test_micro_data_uses_observed_analysis_columns() -> None:
    frame = pl.DataFrame(
        {
            "firm_id": [0, 0, 1],
            "quarter": [1, 2, 1],
            "outcome_reported": [0.0, 1.0, 0.0],
            "perceived_risk": [0.0, 0.2, 0.1],
            "cost_share": [0.1, 0.1, 0.2],
            "neighbor_compliant_share": [0.0, 0.5, 0.0],
            "assoc_compliant_share": [0.0, 0.5, 0.0],
            "privacy_rev_share": [0.2, 0.3, 0.4],
            "phase_phi": [0.0, 0.2, 0.0],
            "compliant_lag": [0.0, 0.0, 0.0],
            "log_size_proxy": [-0.2, -0.2, 0.3],
            "sector": [5, 5, 9],
        }
    )
    data = micro_data_from_frame(frame)
    assert data.n == 3
    assert data.n_sectors == 2
    assert data.sector.tolist() == [0, 0, 1]
    assert not hasattr(data, "capacity")


def test_summary_statistics_contract() -> None:
    aggregate = pl.DataFrame(
        {
            "compliance_rate_obs": [0.1, 0.3, 0.55, 0.7],
            "hhi_obs": [100.0, 110.0, 120.0, 130.0],
            "mean_trust_obs": [0.4, 0.5, 0.6, 0.7],
            "exit_rate_obs": [0.0, 0.01, 0.02, 0.03],
        }
    )
    summary = summary_statistics(aggregate)
    assert summary.shape == (len(SUMMARY_NAMES),)
    np.testing.assert_allclose(summary, [0.7, 3.0, 130.0, 0.55, 0.03, 3.0])


def test_macro_prior_support_and_determinism() -> None:
    first = sample_macro_prior(64, seed=77)
    second = sample_macro_prior(64, seed=77)
    assert first.shape == (64, len(MACRO_PARAMETER_NAMES))
    np.testing.assert_array_equal(first, second)
    assert np.all((first[:, :4] > 0.0) & (first[:, :4] < 1.0))
    assert np.all(first[:, 4:] > 0.0)


def test_tensorized_historical_schedule_has_no_pre_onset_policy_effect() -> None:
    cfg = _config(n_firms=24, n_segments=3, horizon=1)
    world = _world(cfg)
    levers = PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.3)
    active = rollout_tensorized(
        cfg,
        world,
        Theta(),
        levers,
        seed=31,
        quarters=1,
        treatment_start=np.zeros(cfg.population.n_firms, dtype=np.int64),
    )
    future = rollout_tensorized(
        cfg,
        world,
        Theta(),
        levers,
        seed=31,
        quarters=1,
        treatment_start=np.full(cfg.population.n_firms, 8, dtype=np.int64),
    )
    assert float(future.covariates[0]["phase_phi"].max()) == 0.0
    assert float(future.covariates[0]["perceived_risk"].max()) == 0.0
    assert not np.array_equal(
        active.compliance_probabilities.detach().numpy(),
        future.compliance_probabilities.detach().numpy(),
    )


def test_stage_calibration_registers_subprocess_manifest(
    smoke_cfg: RegWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = smoke_cfg
    output_dir = Path(cfg.paths.root) / "calibration"
    output_dir.mkdir(parents=True)
    posterior = output_dir / "posterior.nc"
    posterior.write_bytes(b"durable")
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps({"fitted_parameter_count": 17, "outputs": [str(posterior)]})
    )
    (output_dir / "micro_diagnostics.json").write_text(
        json.dumps({"divergences": 0, "max_r_hat": 1.0, "min_ess_bulk": 500})
    )
    monkeypatch.setattr("regworld.stages._run_script", lambda *args, **kwargs: None)
    outputs = stage_calibration(cfg, NullTracker())
    assert posterior in outputs
    assert output_dir / "calibration_manifest.json" in outputs


def _synthetic_micro_data(n: int, *, seed: int) -> MicroData:
    """Small, moderate-compliance panel for exercising the diagnostics path."""
    rng = np.random.default_rng(seed)
    return MicroData(
        outcome=rng.binomial(1, 0.45, n).astype(np.float32),
        perceived_risk=rng.uniform(0.0, 0.5, n).astype(np.float32),
        cost_share=rng.uniform(0.0, 0.3, n).astype(np.float32),
        neighbor_share=rng.uniform(0.0, 0.6, n).astype(np.float32),
        association_share=rng.uniform(0.0, 0.6, n).astype(np.float32),
        privacy_share=rng.uniform(0.1, 0.5, n).astype(np.float32),
        phase=rng.uniform(0.0, 0.5, n).astype(np.float32),
        compliant_lag=rng.binomial(1, 0.5, n).astype(np.float32),
        log_size=rng.normal(0.0, 0.5, n).astype(np.float32),
        sector=(np.arange(n) % 2).astype(np.int32),
        quarter=(1 + np.arange(n) % 5).astype(np.int32),
        n_sectors=2,
    )


def test_micro_diagnostics_runs_full_arviz_and_energy_path(tmp_path: Path) -> None:
    """Regression guard for the two Stage-4 gate failures (ArviZ adapter + energy field).

    The fast recovery test exercises only the tiny NUTS harness; the real
    diagnostics path (az.summary(kind='all'), az.plot_energy, predictive checks)
    is where both `make calibrate` attempts failed. Run it end to end on a tiny
    synthetic panel so the fix is covered before the full smoke gate.
    """
    train = _synthetic_micro_data(60, seed=101)
    heldout = _synthetic_micro_data(30, seed=202)
    idata, mcmc = fit_micro_numpyro(train, seed=7, warmup=80, draws=80, chains=2)
    output_dir = tmp_path / "calibration"
    payload, paths = run_micro_diagnostics(
        idata, mcmc, train, heldout, seed=7, output_dir=output_dir
    )
    # az.plot_energy only works when the sampler collected the `energy` field.
    assert (output_dir / "energy.png").is_file()
    # az.summary(kind="all") with hdi_prob=0.90 must yield the parsed rows.
    assert (output_dir / "micro_posterior_summary.csv").is_file()
    assert set(payload["parameters"]) == set(MICRO_PARAMETER_NAMES)
    assert isinstance(payload["divergences"], int)
    assert np.isfinite(payload["max_r_hat"])
    assert payload["prior_predictive"]["passed"]
    assert len(paths) == 6


def test_tiny_parameter_recovery_covers_at_least_two_of_three() -> None:
    """PLAN fast gate: 60 designs, 200 draws, at least 2/3 90% coverage."""
    rng = np.random.default_rng(913)
    design = np.column_stack([rng.uniform(0.0, 1.5, 60), rng.uniform(0.0, 1.0, 60)])
    truth = {"beta_0": -1.2, "beta_enforce": 2.5, "beta_peer": 1.4}
    eta = truth["beta_0"] + truth["beta_enforce"] * design[:, 0] + truth["beta_peer"] * design[:, 1]
    probability = 1.0 / (1.0 + np.exp(-eta))
    trials = 12
    successes = rng.binomial(trials, probability)
    idata = fit_tiny_numpyro(
        design,
        successes,
        trials=trials,
        seed=914,
        warmup=200,
        draws=200,
        chains=2,
    )
    covered = 0
    for name, value in truth.items():
        interval = np.quantile(np.asarray(idata.posterior[name]), [0.05, 0.95])
        covered += int(interval[0] <= value <= interval[1])
    assert covered >= 2, f"90% intervals covered {covered}/3 planted parameters"
