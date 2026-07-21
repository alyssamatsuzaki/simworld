"""SALib Morris screening → Sobol first/total-order indices on the GraphRSSM emulator.

The objective J is the episode return (sum of regulator rewards) from a 24-quarter rollout
in EmulatorEnv under a constant lever vector. Morris prunes the 4 factors (always all of them
for now) by effect size; Sobol quantifies S1 (first-order) and ST (total-order) indices.

The emulator-vs-ABM cross-check evaluates a subsample of Sobol design points on the
true ABM (tensorized) and computes the correlation of J across the subsample, confirming
the sensitivity surface agrees.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample

from regworld import rules
from regworld.abm.model import load_observed_world
from regworld.abm.tensorized import rollout_tensorized
from regworld.environments.emulator_env import EmulatorEnv
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.training.datamodule import ACTION_HIGH, ACTION_LOW, load_theta_draws
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Sensitivity analysis output: indices, summary, and metrics."""

    indices: Path  # artifacts/sensitivity/indices.json
    summary: Path  # artifacts/sensitivity/sensitivity_summary.json
    metrics: dict[str, float]


def _objective(
    env: EmulatorEnv,
    cfg: RegWorldConfig,
    action: NDArray[np.float32],
    seed: int,
) -> float:
    """Evaluate J (episode return) under a constant lever vector in the emulator."""
    env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(cfg.horizon_quarters):
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return float(total_reward)


def _salib_problem() -> dict[str, object]:
    """SALib problem dict for the 4 policy levers."""
    return {
        "num_vars": 4,
        "names": ["enforcement", "targeting", "phase_speed", "subsidy"],
        "bounds": [
            [float(ACTION_LOW[0]), float(ACTION_HIGH[0])],  # enforcement: [0, 1]
            [float(ACTION_LOW[1]), float(ACTION_HIGH[1])],  # targeting: [-1, 1]
            [float(ACTION_LOW[2]), float(ACTION_HIGH[2])],  # phase_speed: [0, 1]
            [float(ACTION_LOW[3]), float(ACTION_HIGH[3])],  # subsidy: [0, 1]
        ],
    }


def run_screening(
    cfg: RegWorldConfig,
) -> dict[str, object]:
    """Morris screening to rank the 4 factors by effect size.

    Returns a dict with:
    - morris_mu: mean effect
    - morris_sigma: standard deviation of effect
    - morris_mu_star: mean absolute effect (rank)
    - count: number of trajectories = 8 * (num_vars + 1) = 40
    """
    log.info(
        "Starting Morris screening (D+1 method, %d trajectories)",
        cfg.sensitivity.morris_trajectories,
    )
    problem = _salib_problem()
    samples = morris_sample(
        problem,
        N=cfg.sensitivity.morris_trajectories,
        num_levels=4,
        optimal_trajectories=None,
        seed=cfg.seed,
    )
    log.info("Morris sampled %d design points", samples.shape[0])

    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    env = EmulatorEnv(cfg, model=model, meta=meta)

    outputs = []
    for i, sample in enumerate(samples):
        action = sample.astype(np.float32)
        J = _objective(env, cfg, action, seed=cfg.seed + 1000 + i)
        outputs.append(J)
        if (i + 1) % max(1, len(samples) // 4) == 0:
            log.info("Morris: %d/%d evaluated", i + 1, len(samples))

    outputs_array = np.array(outputs)
    morris_result = morris_analyze(problem, samples, outputs_array, seed=cfg.seed)

    names_list: list[str] = cast(list[str], problem["names"])
    result: dict[str, object] = {
        "method": "Morris",
        "count": len(samples),
        "morris_mu": {
            str(name): float(mu) for name, mu in zip(names_list, morris_result["mu"], strict=True)
        },
        "morris_sigma": {
            str(name): float(sig)
            for name, sig in zip(names_list, morris_result["sigma"], strict=True)
        },
        "morris_mu_star": {
            str(name): float(mu_star)
            for name, mu_star in zip(names_list, morris_result["mu_star"], strict=True)
        },
    }
    log.info("Morris screening done: top drivers by mu_star: %s", result["morris_mu_star"])
    return result


def run_sobol(
    cfg: RegWorldConfig,
) -> tuple[dict[str, object], NDArray[np.float64], NDArray[np.float64]]:
    """Sobol first-order (S1) and total-order (ST) indices on the emulator.

    Saltelli sampling gives (2*D+2)*N evaluations, where D=4 and N=sobol_n.
    Returns the result dict, samples, and outputs for ABM cross-check.
    """
    log.info("Starting Sobol analysis (N=%d, Saltelli sampling)", cfg.sensitivity.sobol_n)
    problem = _salib_problem()
    samples = sobol_sample(
        problem,
        N=cfg.sensitivity.sobol_n,
        calc_second_order=False,
        seed=cfg.seed,
    )
    log.info("Sobol sampled %d design points", samples.shape[0])

    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    env = EmulatorEnv(cfg, model=model, meta=meta)

    outputs = []
    for i, sample in enumerate(samples):
        action = sample.astype(np.float32)
        J = _objective(env, cfg, action, seed=cfg.seed + 10000 + i)
        outputs.append(J)
        if (i + 1) % max(1, len(samples) // 4) == 0:
            log.info("Sobol: %d/%d evaluated", i + 1, len(samples))

    outputs_array = np.array(outputs)
    sobol_result = sobol_analyze(problem, outputs_array, seed=cfg.seed, calc_second_order=False)

    names: list[str] = cast(list[str], problem["names"])
    result: dict[str, object] = {
        "method": "Sobol",
        "count": len(samples),
        "S1": {str(name): float(s1) for name, s1 in zip(names, sobol_result["S1"], strict=True)},
        "S1_conf": {
            str(name): float(conf)
            for name, conf in zip(names, sobol_result["S1_conf"], strict=True)
        },
        "ST": {str(name): float(st) for name, st in zip(names, sobol_result["ST"], strict=True)},
        "ST_conf": {
            str(name): float(conf)
            for name, conf in zip(names, sobol_result["ST_conf"], strict=True)
        },
    }
    log.info("Sobol indices (S1): %s", result["S1"])
    log.info("Sobol indices (ST): %s", result["ST"])
    return result, samples, outputs_array


def run_abm_cross_check(
    cfg: RegWorldConfig,
    sobol_samples: NDArray[np.float64],
    sobol_outputs: NDArray[np.float64],
) -> dict[str, object]:
    """Validate emulator-vs-ABM agreement on a subsample of Sobol design points.

    Randomly select abm_check_points from the Sobol samples, evaluate them in the
    true ABM (tensorized), and compute Spearman correlation of J.
    """
    log.info(
        "Starting ABM cross-check (%d points from Sobol sample)",
        cfg.sensitivity.abm_check_points,
    )

    rng = np.random.default_rng(cfg.seed + 20000)
    n_check = min(cfg.sensitivity.abm_check_points, len(sobol_samples))
    indices = rng.choice(len(sobol_samples), size=n_check, replace=False)

    world = load_observed_world(cfg)
    theta_rows = load_theta_draws(cfg)
    names = list(rules.Theta.__dataclass_fields__)
    theta = rules.Theta(**dict(zip(names, theta_rows.mean(axis=0).tolist(), strict=True)))

    abm_compliance = []
    for idx, i in enumerate(indices):
        sample = sobol_samples[i]
        lever_schedule = np.tile(sample, (cfg.horizon_quarters, 1))
        try:
            truth_run = rollout_tensorized(
                cfg,
                world,
                theta,
                rules.PolicyLevers(),
                seed=cfg.seed + 20000 + idx,
                quarters=cfg.horizon_quarters,
                lever_schedule=lever_schedule,
            )
            final_compliance = float(truth_run.outcomes[-1].compliance_rate.item())
        except Exception as e:
            log.warning("ABM cross-check failed for sample %d: %s", i, e)
            final_compliance = np.nan
        abm_compliance.append(final_compliance)

    abm_array = np.array(abm_compliance)
    emulator_subsample = sobol_outputs[indices]

    valid = ~np.isnan(abm_array) & ~np.isnan(emulator_subsample)
    if valid.sum() < 3:
        corr = None
        log.warning("ABM cross-check: fewer than 3 valid points, skipping correlation")
    else:
        from scipy.stats import spearmanr

        corr_result = spearmanr(emulator_subsample[valid], abm_array[valid])
        corr = float(corr_result.statistic)
        log.info("ABM cross-check: Spearman corr = %.3f (p=%.4f)", corr, corr_result.pvalue)

    return {
        "method": "ABM cross-check (terminal compliance)",
        "sample_size": len(indices),
        "valid_points": int(valid.sum()),
        "emulator_J_mean": float(np.nanmean(emulator_subsample)),
        "abm_compliance_mean": float(np.nanmean(abm_array)),
        "emulator_vs_abm_spearman_corr": corr,
    }


def run_sensitivity(cfg: RegWorldConfig) -> SensitivityResult:
    """Main entry point: Morris screening + Sobol analysis + ABM cross-check."""
    log.info("§14 Sensitivity analysis starting (profile=%s)", cfg.profile_name)

    sensitivity_dir = Path(cfg.paths.root) / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    morris_result = run_screening(cfg)

    sobol_result, sobol_samples, sobol_outputs = run_sobol(cfg)

    abm_result = run_abm_cross_check(cfg, sobol_samples, sobol_outputs)

    indices_obj = {
        "morris": morris_result,
        "sobol": sobol_result,
        "abm_check": abm_result,
    }
    indices_path = sensitivity_dir / "indices.json"
    indices_path.write_text(json.dumps(indices_obj, indent=2))
    log.info("Sensitivity indices → %s", indices_path)

    s1_dict = cast(dict[str, float], sobol_result["S1"])
    summary_obj: dict[str, object] = {
        "profile": cfg.profile_name,
        "seed": cfg.seed,
        "horizon_quarters": cfg.horizon_quarters,
        "methods": ["Morris", "Sobol"],
        "factors": 4,
        "top_driver_by_s1": max(s1_dict.items(), key=lambda x: x[1])[0],
        "top_driver_s1_value": float(max(s1_dict.values())),
        "abm_check_spearman_corr": abm_result["emulator_vs_abm_spearman_corr"],
        "abm_check_sample_size": abm_result["sample_size"],
    }
    summary_path = sensitivity_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2))
    log.info("Sensitivity summary → %s", summary_path)

    metrics: dict[str, float] = {
        "morris_count": float(cast(int, morris_result["count"])),
        "sobol_count": float(cast(int, sobol_result["count"])),
        "abm_check_points": float(cast(int, abm_result["sample_size"])),
        "abm_check_valid": float(cast(int, abm_result["valid_points"])),
        "abm_check_corr": (
            cast(float, abm_result["emulator_vs_abm_spearman_corr"])
            if abm_result["emulator_vs_abm_spearman_corr"] is not None
            else 0.0
        ),
    }

    log.info("§14 Sensitivity analysis done")
    return SensitivityResult(
        indices=indices_path,
        summary=summary_path,
        metrics=metrics,
    )
