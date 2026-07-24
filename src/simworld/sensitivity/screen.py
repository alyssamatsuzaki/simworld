"""SALib sensitivity analysis for Stage 14a — two distinct questions, two designs.

**14a-i · Morris screening of the behavioral parameters θ on the ABM.** This is the
analysis claim C4 rests on ("of ~16 uncertain parameters, a small handful drive most
outcome variance — which tells the client what to measure next"). The factors are the
§7.3 behavioral parameters, sampled inside their prior-derived central intervals, and
each design point is a full tensorized-ABM rollout under the reference policy. Cheap:
`morris_trajectories * (D + 1)` runs. Persisted under the ``morris_theta`` key.

**14a-ii · Morris + Sobol over the four policy levers on the GraphRSSM emulator.** A
different question — how much does each *lever* move the regulator's episode return J —
answered on the emulator because Saltelli sampling needs tens of thousands of rollouts.
Persisted under the ``morris`` and ``sobol`` keys.

The emulator-vs-ABM cross-check evaluates a subsample of Sobol design points on the
true ABM (tensorized) and computes the correlation of J across the subsample, confirming
the sensitivity surface agrees.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample
from scipy.stats import beta as beta_dist
from scipy.stats import halfnorm, norm

from simworld import rules
from simworld.abm.model import load_observed_world
from simworld.abm.tensorized import TensorTrajectory, rollout_tensorized
from simworld.environments.emulator_env import EmulatorEnv
from simworld.training.checkpoint import checkpoint_path, load_checkpoint
from simworld.training.datamodule import ACTION_HIGH, ACTION_LOW, load_theta_draws
from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# §14a-i — the behavioral parameters θ (PLAN §7.3), screened on the ABM
# --------------------------------------------------------------------------------------

# Prior families exactly as tabulated in PLAN §7.3. Only parameters that actually enter
# the estimated ABM's transition equations are screenable:
#   * ``beta_capacity`` multiplies the latent capacity z_i, which is answer-key-only and
#     absent from the fitted model (§7.3) — it is not a factor the client could measure;
#   * ``q0`` / ``q1`` are observation-model nuisance parameters (§7.9); they perturb the
#     *reported* panel, not the dynamics, so an ABM rollout is constant in them.
# That leaves D = 15 of the sixteen §7.3 parameters. Screening the other three would only
# manufacture three guaranteed zeros.
THETA_PRIORS: dict[str, tuple[str, tuple[float, ...]]] = {
    # Group A — firm-decision logit
    "beta_0": ("normal", (0.0, 2.0)),
    "beta_enforce": ("halfnormal", (2.0,)),
    "beta_cost": ("halfnormal", (2.0,)),
    "beta_peer": ("normal", (1.0, 1.0)),
    "beta_assoc": ("normal", (0.5, 1.0)),
    "beta_size": ("normal", (0.0, 1.0)),
    "beta_customer": ("halfnormal", (1.0,)),
    "phi_phase": ("normal", (0.5, 0.5)),
    "beta_stick": ("halfnormal", (1.0,)),
    # Group B — consumer, market, and enforcement dynamics
    "gamma_scale": ("beta", (3.0, 3.0)),
    "ell_learn": ("beta", (2.0, 4.0)),
    "alpha_trust": ("beta", (2.0, 5.0)),
    "rho_influence": ("beta", (2.0, 8.0)),
    "mu_privacy": ("halfnormal", (1.0,)),
    "delta_exit": ("halfnormal", (0.5,)),
}

THETA_NAMES: tuple[str, ...] = tuple(THETA_PRIORS)

# Central prior mass covered by each factor's Morris box.
THETA_BOUND_QUANTILE = 0.05

# Aggregate outcomes screened, per PLAN Stage 14a.
THETA_OUTCOMES: tuple[str, ...] = (
    "terminal_compliance",
    "delta_hhi",
    "exit_rate_cum",
    "terminal_trust",
)


def _prior_interval(family: str, params: tuple[float, ...], q: float) -> tuple[float, float]:
    """Central (1 - 2q) interval of a PLAN §7.3 prior."""
    if family == "normal":
        loc, scale = params
        return float(norm.ppf(q, loc, scale)), float(norm.ppf(1.0 - q, loc, scale))
    if family == "halfnormal":
        (scale,) = params
        return float(halfnorm.ppf(q, scale=scale)), float(halfnorm.ppf(1.0 - q, scale=scale))
    if family == "beta":
        a, b = params
        return float(beta_dist.ppf(q, a, b)), float(beta_dist.ppf(1.0 - q, a, b))
    raise ValueError(f"unknown prior family: {family}")


def theta_bounds() -> list[list[float]]:
    """Prior-derived Morris boxes: the central 90% interval of each §7.3 prior."""
    return [
        list(_prior_interval(family, params, THETA_BOUND_QUANTILE))
        for family, params in THETA_PRIORS.values()
    ]


def _theta_problem() -> dict[str, object]:
    """SALib problem dict for the screenable behavioral parameters θ (§7.3)."""
    return {
        "num_vars": len(THETA_NAMES),
        "names": list(THETA_NAMES),
        "bounds": theta_bounds(),
    }


def _theta_from_vector(base: rules.Theta, values: NDArray[np.float64]) -> rules.Theta:
    """Bind one Morris design row onto a Theta, leaving unscreened fields at ``base``."""
    return replace(
        base,
        **{name: float(value) for name, value in zip(THETA_NAMES, values, strict=True)},
    )


def _by_name(values: NDArray[np.float64] | list[float]) -> dict[str, float]:
    """Zip a SALib per-factor array back onto the θ names."""
    return {name: float(value) for name, value in zip(THETA_NAMES, np.asarray(values), strict=True)}


def _reference_policy(cfg: SimWorldConfig) -> rules.PolicyLevers:
    """Hold the levers at the configured static program while θ varies."""
    return rules.PolicyLevers(
        enforcement=float(cfg.policy.enforcement),
        targeting=float(cfg.policy.targeting),
        phase_speed=float(cfg.policy.phase_speed),
        subsidy=float(cfg.policy.subsidy),
    )


def _theta_outcomes(trajectory: TensorTrajectory) -> dict[str, float]:
    """Extract the four Stage-14a aggregate outcomes from a tensorized rollout."""
    if not trajectory.outcomes:
        return dict.fromkeys(THETA_OUTCOMES, float("nan"))
    first, last = trajectory.outcomes[0], trajectory.outcomes[-1]
    return {
        "terminal_compliance": float(last.compliance_rate.item()),
        "delta_hhi": float(last.hhi.item()) - float(first.hhi.item()),
        "exit_rate_cum": float(last.exit_rate_cum.item()),
        "terminal_trust": float(last.mean_trust.item()),
    }


def run_theta_screen(cfg: SimWorldConfig) -> dict[str, object]:
    """Morris screening of the §7.3 behavioral parameters θ on the tensorized ABM.

    This is the analysis behind claim C4: which of the uncertain behavioral parameters
    actually move the outcomes, and therefore which the client should spend money
    measuring. Every design point is a real ABM rollout — no emulator anywhere.

    Returns the ``morris_theta`` payload (see ``run_sensitivity`` for the JSON shape).
    """
    problem = _theta_problem()
    n_traj = max(2, int(cfg.sensitivity.morris_trajectories))
    log.info(
        "Morris θ-screen on the ABM: D=%d factors, %d trajectories, %d outcomes",
        len(THETA_NAMES),
        n_traj,
        len(THETA_OUTCOMES),
    )
    samples = morris_sample(problem, N=n_traj, num_levels=4, seed=cfg.seed + 30_000)
    log.info("Morris θ-screen sampled %d design points", samples.shape[0])

    world = load_observed_world(cfg)
    base_theta = rules.Theta()
    policy = _reference_policy(cfg)

    responses = np.full((samples.shape[0], len(THETA_OUTCOMES)), np.nan, dtype=np.float64)
    failed = 0
    for i, row in enumerate(samples):
        try:
            trajectory = rollout_tensorized(
                cfg,
                world,
                _theta_from_vector(base_theta, row),
                policy,
                seed=cfg.seed + 30_000 + i,
                quarters=cfg.horizon_quarters,
            )
            values = _theta_outcomes(trajectory)
        except Exception as exc:  # a divergent θ draw must not kill the screen
            log.warning("Morris θ-screen: rollout %d failed (%s)", i, exc)
            failed += 1
            continue
        responses[i] = [values[name] for name in THETA_OUTCOMES]
        if (i + 1) % max(1, samples.shape[0] // 4) == 0:
            log.info("Morris θ-screen: %d/%d rollouts", i + 1, samples.shape[0])

    if failed:
        log.warning("Morris θ-screen: %d/%d rollouts failed", failed, samples.shape[0])
    finite = np.isfinite(responses).all(axis=1)
    if not finite.all():
        # SALib's elementary-effect algebra needs a full design; impute the failures with
        # the column mean so a single bad draw degrades rather than destroys the screen.
        log.warning("Morris θ-screen: imputing %d non-finite responses", int((~finite).sum()))
        column_means = np.nanmean(responses[finite], axis=0) if finite.any() else np.zeros(4)
        responses = np.where(np.isfinite(responses), responses, column_means)

    per_outcome: dict[str, object] = {}
    mu_star_shares = np.zeros((len(THETA_OUTCOMES), len(THETA_NAMES)), dtype=np.float64)
    for j, outcome_name in enumerate(THETA_OUTCOMES):
        analysis = morris_analyze(
            problem, samples, responses[:, j], num_levels=4, seed=cfg.seed + 30_000
        )
        mu_star = np.asarray(analysis["mu_star"], dtype=np.float64)
        total = float(mu_star.sum())
        mu_star_shares[j] = mu_star / total if total > 0.0 else 0.0
        order = np.argsort(-mu_star)
        per_outcome[outcome_name] = {
            "mu": _by_name(analysis["mu"]),
            "mu_star": _by_name(mu_star),
            "sigma": _by_name(analysis["sigma"]),
            "mu_star_conf": _by_name(analysis["mu_star_conf"]),
            "ranking": [THETA_NAMES[k] for k in order],
        }

    mean_share = mu_star_shares.mean(axis=0)
    ranking = [THETA_NAMES[k] for k in np.argsort(-mean_share)]
    top_k = max(1, min(int(cfg.sensitivity.top_k), len(THETA_NAMES)))
    screened_in = ranking[:top_k]

    result: dict[str, object] = {
        "method": "Morris elementary effects on the tensorized ABM",
        "target": "abm",
        "question": "which behavioral parameters θ (§7.3) drive outcome variance (claim C4)",
        "num_vars": len(THETA_NAMES),
        "names": list(THETA_NAMES),
        "bounds": {
            name: [float(lo), float(hi)]
            for name, (lo, hi) in zip(THETA_NAMES, theta_bounds(), strict=True)
        },
        "bound_quantile": THETA_BOUND_QUANTILE,
        "trajectories": n_traj,
        "num_levels": 4,
        "count": int(samples.shape[0]),
        "failed_runs": failed,
        "policy": list(policy.as_array().astype(float)),
        "outcomes": list(THETA_OUTCOMES),
        "per_outcome": per_outcome,
        "mu_star_share_mean": {
            name: float(value) for name, value in zip(THETA_NAMES, mean_share, strict=True)
        },
        "ranking": ranking,
        "top_k": top_k,
        "screened_in": screened_in,
        "screened_out": ranking[top_k:],
    }
    log.info("Morris θ-screen: top %d drivers = %s", top_k, screened_in)
    return result


@dataclass
class SensitivityResult:
    """Sensitivity analysis output: indices, summary, and metrics."""

    indices: Path  # artifacts/sensitivity/indices.json
    summary: Path  # artifacts/sensitivity/sensitivity_summary.json
    metrics: dict[str, float]


def _objective(
    env: EmulatorEnv,
    cfg: SimWorldConfig,
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
    cfg: SimWorldConfig,
) -> dict[str, object]:
    """Morris screening of the 4 *policy levers* on the emulator, ranked by effect size.

    Distinct from :func:`run_theta_screen`, which screens the behavioral parameters θ on
    the ABM and is what claim C4 rests on. This one ranks the levers by their effect on
    the regulator's episode return J.

    Returns a dict with:
    - morris_mu: mean effect
    - morris_sigma: standard deviation of effect
    - morris_mu_star: mean absolute effect (rank)
    - count: number of design points = morris_trajectories * (num_vars + 1)
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
    cfg: SimWorldConfig,
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


def _abm_episode_return(cfg: SimWorldConfig, truth_run: TensorTrajectory) -> float:
    """Episode return J from a tensorized ABM rollout, matching the emulator's J.

    The emulator's Sobol objective sums ``regulator_reward`` over the episode when
    ``emulator.reward_from_outcomes`` (the ABM has no reward head, so the ABM J is
    always the recomputed reward). Same weights, same baseline convention as
    ``EmulatorEnv.step`` — this makes the cross-check a like-for-like J comparison.
    """
    weights = cast(
        tuple[float, float, float, float, float, float],
        tuple(
            float(getattr(cfg.objective, name))
            for name in ("w_c", "w_h", "w_s", "w_e", "w_t", "w_x")
        ),
    )
    constants = rules.Constants()

    def _to_outcome(o: object) -> rules.QuarterOutcome:
        terc = np.asarray(o.compliance_by_tercile.detach()).ravel().tolist()  # type: ignore[attr-defined]
        return rules.QuarterOutcome(
            compliance_rate=float(o.compliance_rate.item()),  # type: ignore[attr-defined]
            compliance_rate_weighted=float(o.compliance_rate_weighted.item()),  # type: ignore[attr-defined]
            compliance_by_tercile=(float(terc[0]), float(terc[1]), float(terc[2])),
            hhi=float(o.hhi.item()),  # type: ignore[attr-defined]
            mean_trust=float(o.mean_trust.item()),  # type: ignore[attr-defined]
            consumer_surplus=float(o.consumer_surplus.item()),  # type: ignore[attr-defined]
            exit_rate_cum=float(o.exit_rate_cum.item()),  # type: ignore[attr-defined]
            enforcement_cost=float(o.enforcement_cost.item()),  # type: ignore[attr-defined]
            n_audits=int(o.n_audits.item()),  # type: ignore[attr-defined]
        )

    baseline = _to_outcome(truth_run.outcomes[0])
    return float(
        sum(
            rules.regulator_reward(
                _to_outcome(o), baseline, weights, constants, cfg.population.n_firms
            )
            for o in truth_run.outcomes
        )
    )


def run_abm_cross_check(
    cfg: SimWorldConfig,
    sobol_samples: NDArray[np.float64],
    sobol_outputs: NDArray[np.float64],
) -> dict[str, object]:
    """Validate emulator-vs-ABM agreement on a subsample of Sobol design points.

    Randomly select abm_check_points from the Sobol samples, evaluate the SAME
    objective J (summed regulator reward) in the true ABM (tensorized) as the
    emulator Sobol run computed, and report the Spearman rank correlation and the
    mean absolute J gap. Reported, not gated: a low correlation means the Sobol
    indices inherited a wrong emulator and the report says so (§14a guard).
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

    abm_returns = []
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
            abm_j = _abm_episode_return(cfg, truth_run)
        except Exception as e:
            log.warning("ABM cross-check failed for sample %d: %s", i, e)
            abm_j = np.nan
        abm_returns.append(abm_j)

    abm_array = np.array(abm_returns)
    emulator_subsample = sobol_outputs[indices]

    valid = ~np.isnan(abm_array) & ~np.isnan(emulator_subsample)
    mean_abs_gap = (
        float(np.mean(np.abs(emulator_subsample[valid] - abm_array[valid])))
        if valid.sum()
        else float("nan")
    )
    if valid.sum() < 3:
        corr = None
        log.warning("ABM cross-check: fewer than 3 valid points, skipping correlation")
    else:
        from scipy.stats import spearmanr

        corr_result = spearmanr(emulator_subsample[valid], abm_array[valid])
        corr = float(corr_result.statistic)
        log.info(
            "ABM cross-check: J Spearman corr = %.3f (p=%.4f), mean|dJ| = %.3f",
            corr,
            corr_result.pvalue,
            mean_abs_gap,
        )

    return {
        "method": "ABM cross-check (episode return J, like-for-like)",
        "sample_size": len(indices),
        "valid_points": int(valid.sum()),
        "emulator_J_mean": float(np.nanmean(emulator_subsample)),
        "abm_J_mean": float(np.nanmean(abm_array)),
        "emulator_vs_abm_J_mean_abs_gap": mean_abs_gap,
        "emulator_vs_abm_spearman_corr": corr,
    }


def run_sensitivity(cfg: SimWorldConfig) -> SensitivityResult:
    """Main entry point: θ Morris screen on the ABM, then levers on the emulator.

    Writes ``indices.json`` with four top-level keys:

    - ``morris_theta`` — Stage 14a-i, Morris elementary effects over the §7.3 behavioral
      parameters θ evaluated on the tensorized ABM. **This is claim C4.**
    - ``morris`` — Morris over the 4 policy levers on the emulator (J = episode return).
    - ``sobol`` — Saltelli/Sobol S1 and ST over the same 4 levers on the emulator.
    - ``abm_check`` — emulator-vs-ABM agreement on a Sobol subsample.
    """
    log.info("§14 Sensitivity analysis starting (profile=%s)", cfg.profile_name)

    sensitivity_dir = Path(cfg.paths.root) / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    theta_result = run_theta_screen(cfg)

    morris_result = run_screening(cfg)

    sobol_result, sobol_samples, sobol_outputs = run_sobol(cfg)

    abm_result = run_abm_cross_check(cfg, sobol_samples, sobol_outputs)

    indices_obj = {
        "morris_theta": theta_result,
        "morris": morris_result,
        "sobol": sobol_result,
        "abm_check": abm_result,
    }
    indices_path = sensitivity_dir / "indices.json"
    indices_path.write_text(json.dumps(indices_obj, indent=2))
    log.info("Sensitivity indices → %s", indices_path)

    s1_dict = cast(dict[str, float], sobol_result["S1"])
    theta_ranking = cast(list[str], theta_result["ranking"])
    summary_obj: dict[str, object] = {
        "profile": cfg.profile_name,
        "seed": cfg.seed,
        "horizon_quarters": cfg.horizon_quarters,
        "methods": ["Morris (θ on ABM)", "Morris (levers on emulator)", "Sobol (levers)"],
        "factors": 4,
        "theta_factors": theta_result["num_vars"],
        "theta_top_drivers": theta_result["screened_in"],
        "theta_top_driver": theta_ranking[0] if theta_ranking else None,
        "top_driver_by_s1": max(s1_dict.items(), key=lambda x: x[1])[0],
        "top_driver_s1_value": float(max(s1_dict.values())),
        "abm_check_spearman_corr": abm_result["emulator_vs_abm_spearman_corr"],
        "abm_check_sample_size": abm_result["sample_size"],
    }
    summary_path = sensitivity_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2))
    log.info("Sensitivity summary → %s", summary_path)

    metrics: dict[str, float] = {
        "morris_theta_factors": float(cast(int, theta_result["num_vars"])),
        "morris_theta_count": float(cast(int, theta_result["count"])),
        "morris_theta_failed": float(cast(int, theta_result["failed_runs"])),
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
