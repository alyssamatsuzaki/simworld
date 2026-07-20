"""Surrogate-assisted SMC-ABC for aggregate-only behavioral parameters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import beta as beta_dist
from scipy.stats import halfnorm, qmc
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from regworld.abm.model import load_observed_world
from regworld.abm.tensorized import rollout_tensorized
from regworld.calibration.summaries import SUMMARY_NAMES, summary_statistics
from regworld.rules import PolicyLevers, Theta
from regworld.types import RegWorldConfig

MACRO_PARAMETER_NAMES = (
    "gamma_scale",
    "ell_learn",
    "alpha_trust",
    "rho_influence",
    "mu_privacy",
    "delta_exit",
)


def _prior_transform(unit: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    u = np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)
    return np.column_stack(
        [
            beta_dist.ppf(u[:, 0], 3.0, 3.0),
            beta_dist.ppf(u[:, 1], 2.0, 4.0),
            beta_dist.ppf(u[:, 2], 2.0, 5.0),
            beta_dist.ppf(u[:, 3], 2.0, 8.0),
            halfnorm.ppf(u[:, 4], scale=1.0),
            halfnorm.ppf(u[:, 5], scale=0.5),
        ]
    )


def sample_macro_prior(n: int, seed: int) -> np.ndarray:
    """Stratified draws from the six priors in PLAN §7.3."""
    if n < 1:
        raise ValueError("number of prior draws must be positive")
    sampler = qmc.LatinHypercube(d=len(MACRO_PARAMETER_NAMES), seed=seed)
    return _prior_transform(sampler.random(n))


def _policy(cfg: RegWorldConfig) -> PolicyLevers:
    del cfg
    # Published settings of the historical program used to build the observed
    # panel. Forecast-policy settings are intentionally not used for calibration.
    return PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.3)


def _observed_treatment_start(cfg: RegWorldConfig, firm_region: np.ndarray) -> np.ndarray:
    """Reconstruct the historical staggered rollout from the sampled panel.

    Treatment quarters are public program metadata on every sampled row. Missing
    firms use their observed-world region proxy; no concealed firm state is read.
    """
    panel = pl.read_parquet(Path(cfg.paths.data) / "observed" / "firm_panel.parquet")
    cohorts = panel.select("region", "treatment_quarter").unique()
    by_region = {
        int(row["region"]): int(row["treatment_quarter"]) for row in cohorts.iter_rows(named=True)
    }
    sentinel = cfg.horizon_quarters + cfg.observed_quarters + 1
    starts = np.asarray(
        [by_region.get(int(region), sentinel) for region in firm_region], dtype=np.int64
    )
    return np.where(starts > 0, starts - 1, sentinel)


def _theta(base: Theta, values: np.ndarray) -> Theta:
    return replace(base, **dict(zip(MACRO_PARAMETER_NAMES, values, strict=True)))


def build_tensor_design(
    cfg: RegWorldConfig,
    *,
    base_theta: Theta,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Generate a simulation design with the sparse tensorized ABM.

    Replicate summaries are averaged at each Latin-hypercube point. The fitted
    tree ensemble is subsequently only an interpolation layer; every training
    target in it came from the full firm/consumer transition system.
    """
    world = load_observed_world(cfg, seed=cfg.seed)
    treatment_start = _observed_treatment_start(cfg, world.firms.region)
    design = sample_macro_prior(cfg.calibration.design_points, cfg.seed + 4_100)
    summaries = np.empty((design.shape[0], len(SUMMARY_NAMES)), dtype=np.float64)
    replicates = max(1, int(cfg.calibration.replicates))
    for index, values in enumerate(design):
        runs = []
        for replicate in range(replicates):
            trajectory = rollout_tensorized(
                cfg,
                world,
                _theta(base_theta, values),
                _policy(cfg),
                seed=cfg.seed + 50_000 + 1009 * index + replicate,
                quarters=cfg.observed_quarters,
                treatment_start=treatment_start,
            )
            runs.append(summary_statistics(trajectory))
        summaries[index] = np.mean(runs, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "macro_tensor_design.npz"
    np.savez_compressed(
        path,
        parameters=design,
        summaries=summaries,
        parameter_names=np.asarray(MACRO_PARAMETER_NAMES),
        summary_names=np.asarray(SUMMARY_NAMES),
        replicates=np.asarray([replicates]),
    )
    return design, summaries, path


def _fit_surrogate(
    design: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[ExtraTreesRegressor, dict[str, Any]]:
    # A deterministic holdout quantifies the interpolation error instead of
    # treating the scalable approximation as if it were the simulator itself.
    rng = np.random.default_rng(seed)
    order = rng.permutation(design.shape[0])
    n_test = max(1, design.shape[0] // 5)
    test = order[:n_test]
    train = order[n_test:]
    if train.size < 8:
        train = order
        test = order
    model = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=2 if train.size >= 16 else 1,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(design[train], targets[train])
    prediction = model.predict(design[test])
    mae = mean_absolute_error(targets[test], prediction, multioutput="raw_values")
    # Refit on every expensive design point after measuring honest holdout error.
    model.fit(design, targets)
    return model, {
        "backend": "extra_trees_over_tensorized_design",
        "n_design_points": int(design.shape[0]),
        "holdout_points": int(test.size),
        "holdout_mae": dict(zip(SUMMARY_NAMES, np.asarray(mae).tolist(), strict=True)),
    }


def _distance_scale(aggregate: pl.DataFrame) -> np.ndarray:
    horizon = aggregate.height

    def standard_deviation(name: str) -> float:
        values = aggregate[name].to_numpy().astype(np.float64, copy=False)
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    return np.asarray(
        [
            max(standard_deviation("compliance_rate_obs"), 0.05),
            max(horizon / 4.0, 1.0),
            max(standard_deviation("hhi_obs"), 25.0),
            max(standard_deviation("mean_trust_obs"), 0.05),
            max(standard_deviation("exit_rate_obs"), 0.02),
            max(horizon / 4.0, 1.0),
        ],
        dtype=np.float64,
    )


def _clip_support(particles: np.ndarray) -> np.ndarray:
    out = particles.copy()
    out[:, :4] = np.clip(out[:, :4], 1e-4, 1.0 - 1e-4)
    out[:, 4] = np.clip(out[:, 4], 1e-4, 4.0)
    out[:, 5] = np.clip(out[:, 5], 1e-4, 2.0)
    return out


def fit_macro_smc(
    cfg: RegWorldConfig,
    s_obs: np.ndarray,
    *,
    base_theta: Theta | None = None,
    aggregate: pl.DataFrame | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Fit Group B using adaptive-kernel SMC-ABC and return InferenceData.

    Expensive responses are a tensorized-ABM design. SMC mutations are evaluated
    by an explicitly diagnosed tree surrogate, which is the smoke/dev scaling
    approximation rather than an unlabelled replacement for the simulation.
    """
    import arviz as az

    output_dir = output_dir or Path(cfg.paths.root) / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    if aggregate is None:
        path = Path(cfg.paths.data) / "observed" / "aggregate_series.parquet"
        aggregate = pl.read_parquet(path).sort("quarter").head(cfg.observed_quarters)
    target = np.asarray(s_obs, dtype=np.float64)
    if target.shape != (len(SUMMARY_NAMES),):
        raise ValueError(f"s_obs must have shape ({len(SUMMARY_NAMES)},)")

    design, summaries, design_path = build_tensor_design(
        cfg,
        base_theta=base_theta or Theta(),
        output_dir=output_dir,
    )
    surrogate, metadata = _fit_surrogate(design, summaries, cfg.seed + 4_200)
    scale = _distance_scale(aggregate)
    rng = np.random.default_rng(cfg.seed + 4_300)
    n_particles = int(cfg.calibration.smc_abc.particles)
    particles = sample_macro_prior(n_particles, cfg.seed + 4_400)
    weights = np.full(n_particles, 1.0 / n_particles)
    rounds: list[dict[str, float]] = []
    distances = np.zeros(n_particles, dtype=np.float64)
    for round_index in range(int(cfg.calibration.smc_abc.rounds)):
        prediction = surrogate.predict(particles)
        distances = np.sqrt(np.mean(((prediction - target) / scale) ** 2, axis=1))
        epsilon = max(
            float(np.quantile(distances, cfg.calibration.smc_abc.quantile)),
            1e-6,
        )
        likelihood = np.exp(-0.5 * (distances / epsilon) ** 2)
        weights = likelihood / max(float(likelihood.sum()), 1e-300)
        ess = 1.0 / float(np.sum(weights**2))
        rounds.append({"round": float(round_index + 1), "epsilon": epsilon, "ess": ess})
        if round_index + 1 < cfg.calibration.smc_abc.rounds:
            parent = rng.choice(n_particles, size=n_particles, replace=True, p=weights)
            spread = np.std(particles, axis=0, ddof=1)
            jitter = rng.normal(size=particles.shape) * spread * (0.35 / (round_index + 1))
            particles = _clip_support(particles[parent] + jitter)
            weights.fill(1.0 / n_particles)

    final_index = rng.choice(n_particles, size=n_particles, replace=True, p=weights)
    posterior = particles[final_index]
    posterior_distance = distances[final_index]
    idata = az.from_dict(
        posterior={
            name: posterior[:, index][None, :] for index, name in enumerate(MACRO_PARAMETER_NAMES)
        },
        sample_stats={"abc_distance": posterior_distance[None, :]},
    )
    metadata.update(
        {
            "summary_names": list(SUMMARY_NAMES),
            "observed_summary": target.tolist(),
            "distance_scale": scale.tolist(),
            "rounds": rounds,
            "design_artifact": str(design_path),
        }
    )
    (output_dir / "macro_diagnostics.json").write_text(json.dumps(metadata, indent=2))
    idata.to_netcdf(output_dir / "macro_posterior.nc")
    return idata
