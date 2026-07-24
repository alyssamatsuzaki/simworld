"""Stage implementations for the driver. Filled in phase by phase (§10).

Contract: `stage_<name>(cfg, tracker) -> list[Path]` of durable outputs.
Raise `simworld.pipeline.Degraded(note)` for an honest partial result.
Heavy imports stay inside the functions: the driver process must not pay for
(or conflict with) libraries a disabled stage would have used. Calibration is
always launched as a subprocess so JAX never enters this process (§5).

`isolated_envs=true` (§5 fallback): the driver routes every script-backed stage
in STAGE_SCRIPTS through `uv run` inside a per-extras-group venv
(`.venv-<group>`, group "core" when the stage needs no extra), synced once per
group per process. Stages with no script entry point — recon, tensorized_abm,
envs, marl — are pure-core checks and always run in-process regardless: the
driver's own core venv is exactly the environment they need.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from simworld.tracking import Tracker
from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)

# Stage name -> (script entry points, run in order; uv extras group, None = core-only).
STAGE_SCRIPTS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "data": (("generate_world.py", "make_data.py"), None),
    "graphs": (("build_graphs.py",), None),
    "abm": (("run_abm.py",), None),
    "calibration": (("calibrate.py",), "bayes"),
    "causal": (("causal_analysis.py", "validate_simulator.py"), "causal"),
    "emulator": (("train_emulator.py",), None),
    "rl": (("train_rl.py", "train_marl.py"), "rl"),
    "ensemble": (("run_ensemble.py",), "rl"),
    "sensitivity": (("sensitivity.py",), "opt"),
    # The §11 eval suite spans families that need bayes/causal/rl/opt at once, so
    # its isolated venv gets every extra ("all") rather than a single group.
    "evaluation": (("eval_emulator.py",), "all"),
    "figures": (("make_figures.py",), "app"),
    "report": (("build_report.py",), None),
}

# Extras groups whose isolated venv has already been `uv sync`ed in this process.
_SYNCED_GROUPS: set[str] = set()


def _sync_group_env(group: str, env: dict[str, str]) -> None:
    """Create/refresh the per-group venv once per group per process (isolated mode)."""
    if group in _SYNCED_GROUPS:
        return
    if group == "all":
        cmd = ["uv", "sync", "--all-extras", "-q"]
    else:
        cmd = ["uv", "sync", "--extra", "dev"]
        if group != "core":
            cmd += ["--extra", group]
        cmd.append("-q")
    log.info("subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    _SYNCED_GROUPS.add(group)


def _run_script(
    cfg: SimWorldConfig,
    script: str,
    extra_overrides: list[str] | None = None,
    env: dict[str, str] | None = None,
    group: str | None = None,
) -> None:
    """Run a stage script as a subprocess with the current profile (JAX isolation, §5).

    With `cfg.isolated_envs` the script runs via `uv run --no-sync` inside the
    per-group venv `.venv-<group>` ("core" when `group` is None), synced first
    via `_sync_group_env`; otherwise it runs under the current interpreter.
    """
    full_env = dict(os.environ)
    full_env.setdefault("JAX_PLATFORMS", "cpu")
    full_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    full_env.update(env or {})
    overrides = [f"profile={cfg.profile_name}", *(extra_overrides or [])]
    if cfg.isolated_envs:
        group_name = group or "core"
        full_env["UV_PROJECT_ENVIRONMENT"] = f".venv-{group_name}"
        _sync_group_env(group_name, full_env)
        cmd = ["uv", "run", "--no-sync", "python", f"scripts/{script}", *overrides]
    else:
        cmd = [sys.executable, f"scripts/{script}", *overrides]
    log.info("subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=full_env)


def isolated_stage(name: str) -> Callable[[SimWorldConfig, Tracker], list[Path]]:
    """Stage callable running `name`'s scripts in their per-group venv (§5 fallback).

    Isolated stages communicate through files on disk only (§5); they return no
    output paths, so the driver re-runs them rather than CACHED-skipping.
    """
    scripts, group = STAGE_SCRIPTS[name]

    def run(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
        for script in scripts:
            _run_script(cfg, script, group=group)
        return []

    return run


def stage_recon(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Record the installed stack and any skipped extras into the run."""
    skips = Path(".stage_skips")
    skipped = skips.read_text().split() if skips.exists() else []
    out = Path(cfg.paths.root) / "recon.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "polars", "networkx", "mesa", "torch", "gymnasium", "mlflow"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as e:
            versions[mod] = f"unavailable: {type(e).__name__}"
    out.write_text(json.dumps({"skipped_extras": skipped, "versions": versions}, indent=2))
    return [out]


def stage_data(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 1: generate the world, degrade it, ingest the analysis panel (§10)."""
    from simworld.data.duck import build_views
    from simworld.data.generate import generate_ground_truth
    from simworld.data.ingest import ingest

    result = generate_ground_truth(cfg)
    panel_path = ingest(cfg)
    views = build_views(cfg)
    tracker.log_metrics(
        {
            "data_observed_artifacts": len(result.observed_paths),
            "data_sealed_artifacts": len(result.sealed_paths),
        }
    )
    return [*result.observed_paths, *result.sealed_paths, panel_path, views]


def stage_graphs(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 2: observed edges -> metrics + PyG HeteroData (§10)."""
    import json

    import polars as pl
    import torch

    from simworld.data.store import observed_dir
    from simworld.graphs.to_pyg import hetero_from_edges, static_node_features

    gdir = observed_dir(cfg) / "graphs"
    edges = {p.stem: pl.read_parquet(p) for p in sorted(gdir.glob("*.parquet"))}
    registry = pl.read_parquet(observed_dir(cfg) / "firm_registry.parquet")
    survey = pl.read_parquet(observed_dir(cfg) / "consumer_survey.parquet")
    data = hetero_from_edges(cfg, edges, static_node_features(cfg, registry, survey))
    out_dir = Path(cfg.paths.graphs)
    out_dir.mkdir(parents=True, exist_ok=True)
    hetero_path = out_dir / "hetero_observed.pt"
    torch.save(data, hetero_path)
    summary_path = out_dir / "hetero_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "node_types": {k: int(data[k].x.shape[0]) for k in data.node_types},
                "edge_types": {
                    "__".join(et): int(data[et].edge_index.shape[1]) for et in data.edge_types
                },
            },
            indent=2,
        )
    )
    tracker.log_metrics({"graph_edge_types": float(len(data.edge_types))})
    return [hetero_path, summary_path]


def stage_abm(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 3: observed-world Mesa model under the configured policy."""
    from simworld.abm.collect import run_observed_abm

    trajectory, paths = run_observed_abm(cfg, include_tensorized=False)
    terminal = trajectory.outcomes[-1]
    tracker.log_metrics(
        {
            "abm_terminal_compliance": terminal.compliance_rate,
            "abm_terminal_hhi": terminal.hhi,
            "abm_terminal_exit_rate": terminal.exit_rate_cum,
            "abm_quarters": float(len(trajectory.outcomes)),
        }
    )
    return paths


def stage_tensorized_abm(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 3b: differentiable sparse-tensor simulation on observed inputs."""
    import polars as pl

    from simworld.abm.collect import run_tensorized_abm

    paths = run_tensorized_abm(cfg)
    terminal = pl.read_parquet(paths[0]).tail(1).to_dicts()[0]
    tracker.log_metrics(
        {
            "tensor_terminal_compliance": float(terminal["compliance_rate"]),
            "tensor_terminal_hhi": float(terminal["hhi"]),
            "tensor_terminal_exit_rate": float(terminal["exit_rate"]),
        }
    )
    return paths


def stage_calibration(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 4: launch JAX inference out of process and register its artifacts."""
    _run_script(cfg, "calibrate.py", group="bayes")
    output_dir = Path(cfg.paths.root) / "calibration"
    manifest_path = output_dir / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    diagnostics = json.loads((output_dir / "micro_diagnostics.json").read_text())
    tracker.log_metrics(
        {
            "calibration_divergences": float(diagnostics["divergences"]),
            "calibration_max_rhat": float(diagnostics["max_r_hat"]),
            "calibration_min_ess_bulk": float(diagnostics["min_ess_bulk"]),
            "calibration_parameters": float(manifest["fitted_parameter_count"]),
        }
    )
    outputs = [manifest_path, *[Path(path) for path in manifest["outputs"]]]
    if cfg.calibration.recovery_grid:
        # C1 is a two-world contrast; a single run only ships one variant, so the
        # recovery grid re-calibrates under both wellspecified and confounded worlds
        # (dev-gated — off at smoke for the < 6 min budget).
        _run_script(cfg, "recovery_grid.py", group="bayes")
        grid_path = output_dir / "recovery_grid.json"
        if grid_path.is_file():
            contrast = json.loads(grid_path.read_text()).get("contrast", {})
            tracker.log_metrics({"c1_clean_contrast": float(bool(contrast.get("clean_contrast")))})
            outputs.append(grid_path)
    return outputs


def stage_causal(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 5: causal estimates, refutation, discovery, and the four-number gate."""
    from simworld.causal.gate import run_gate, write_gate_outputs
    from simworld.pipeline import Degraded

    _run_script(cfg, "causal_analysis.py", group="causal")
    estimates_path = Path(cfg.paths.root) / "causal" / "causal_estimates.json"
    estimates = json.loads(estimates_path.read_text())
    result = run_gate(cfg)
    gate_paths = write_gate_outputs(cfg, result)
    tracker.log_metrics(
        {
            "causal_tau_true": result.tau_true,
            "causal_tau_abm": result.tau_abm,
            "causal_tau_qe": result.tau_qe,
            "causal_tau_obs": result.tau_obs,
            "causal_gate_flagged": float(result.flagged),
            "causal_did_placebo": float(estimates["refutation"]["placebo_effect"]),
        }
    )
    paths = [estimates_path, *gate_paths]
    if result.flagged:
        # §10 5f on_disagreement: surface the discrepancy loudly; the recalibration
        # pass is a driver-level decision, not something to hide inside this stage.
        raise Degraded(
            f"simulator gate FLAGGED (see {gate_paths[-1]}); "
            f"causal.on_disagreement={cfg.causal.on_disagreement}",
            outputs=[str(p) for p in paths],
        )
    return paths


def stage_emulator(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stages 6+7: train the GraphRSSM and certify the EmulatorEnv contract half."""
    import numpy as np

    from simworld.environments.emulator_env import EmulatorEnv
    from simworld.training.train_emulator import train_world_model

    result = train_world_model(cfg)
    tracker.log_metrics(
        {
            "emulator_val_total": result.metrics["val_total"],
            "emulator_val_aggregate": result.metrics["val_aggregate"],
            "emulator_val_imag_aggregate": result.metrics["val_imag_aggregate"],
            "emulator_train_seconds": result.metrics["train_seconds"],
            "emulator_parameters": result.metrics["parameters"],
        }
    )
    env = EmulatorEnv(cfg)
    observation, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    contract = Path(cfg.paths.root) / "envs" / "emulator_contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(
        json.dumps(
            {
                "backend": "emulator",
                "arch": result.arch,
                "observation_shape": list(observation.shape),
                "action_shape": list(env.action_space.shape or ()),
                "reward_finite": bool(np.isfinite(reward)),
                "terminated": terminated,
                "truncated": truncated,
            },
            indent=2,
        )
    )
    env.close()
    return [result.checkpoint, result.summary, contract]


def stage_envs(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 8 checkpoint: validate and record the current Gymnasium contract."""
    from simworld.environments.abm_env import AbmEnv

    env = AbmEnv(cfg)
    observation, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    out = Path(cfg.paths.root) / "envs" / "abm_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "backend": "mesa",
                "observation_shape": list(observation.shape),
                "action_shape": list(env.action_space.shape or ()),
                "reward_finite": bool(float("-inf") < reward < float("inf")),
                "terminated": terminated,
                "truncated": truncated,
            },
            indent=2,
        )
    )
    tracker.log_metrics({"env_observation_dims": float(observation.size)})
    env.close()
    return [out]


def stage_marl(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 9 checkpoint: exercise one strategic Parallel-API transition."""
    from simworld.environments.marl_env import RegulationMARLEnv

    env = RegulationMARLEnv(cfg)
    observations, _ = env.reset(seed=cfg.seed)
    actions = {
        agent: env.action_space(agent).low.copy()  # type: ignore[attr-defined]
        for agent in env.agents
    }
    _, rewards, terminations, truncations, _ = env.step(actions)
    out = Path(cfg.paths.root) / "envs" / "marl_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "agents": list(observations),
                "n_agents": len(observations),
                "rewards_finite": all(
                    float("-inf") < value < float("inf") for value in rewards.values()
                ),
                "terminations": terminations,
                "truncations": truncations,
            },
            indent=2,
        )
    )
    tracker.log_metrics({"marl_agents": float(len(observations))})
    env.close()
    return [out]


def stage_rl(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 10 (+10d): SB3 PPO and the latent Dreamer actor-critic against the
    emulator, then the strategic-firm MARL ablation that makes claim C6 computable.

    The Stage-10d ablation (IPPO by iterated best response) writes
    ``artifacts/marl/c6_comparison.json`` — the artifact ``evaluation.report``
    reads to answer C6, which is otherwise "unanswered" at every scale because
    nothing in the driver produced it. At smoke budgets (``rl.marl_timesteps``)
    the strategic firms are undertrained and the report says so honestly;
    ``profile=dev`` runs the full ~200k-timestep ablation for a real verdict.
    The ablation is non-gating (PLAN.md guardrail 11, DEGRADED by design), so a
    failure inside it leaves C6 unanswered rather than sinking Stage 10.
    """
    from simworld.agents import train_rl

    result = train_rl(cfg)
    tracker.log_metrics(result.metrics)
    outputs = [*result.checkpoints, result.summary]

    try:
        from simworld.agents.marl import train_marl

        marl_result = train_marl(cfg)
        tracker.log_metrics(marl_result.metrics)
        outputs.extend([marl_result.comparison, marl_result.summary])
    except Exception as exc:  # Stage 10d is non-gating; keep C6 unanswered, don't fail Stage 10
        log.warning("Stage 10d MARL ablation failed (%s); C6 will remain unanswered", exc)

    return outputs


def stage_ensemble(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 11: Ray-scalable scenario cube + ABM cross-validation.

    The Phase-6 acceptance gate (§10 Stage 11, §18) is coverage >= 0.85 against the ABM
    cross-check. It is enforced *after* the cube and summary are written, so a failing
    gate marks the stage FAILED without destroying the artifacts needed to diagnose it.
    Ungated at profile=smoke, where too few cells make the number meaningless.
    """
    from simworld.ensemble import run_ensemble
    from simworld.ensemble.validation import enforce_coverage_gate

    result = run_ensemble(cfg)
    tracker.log_metrics(result.metrics)
    enforce_coverage_gate(cfg, float(result.metrics["coverage"]))
    return [result.cube, result.summary]


def stage_sensitivity(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 14: SALib Morris -> Sobol sensitivity on the emulator + Optuna policy search."""
    from simworld.sensitivity.policy_search import run_policy_search, save_optuna_best
    from simworld.sensitivity.screen import run_sensitivity

    result = run_sensitivity(cfg)
    tracker.log_metrics(result.metrics)

    optuna_result = run_policy_search(cfg)
    optuna_path = save_optuna_best(cfg, optuna_result)
    tracker.log_metrics({"sensitivity_optuna_best_J": float(optuna_result["best_J"])})  # type: ignore[arg-type]

    return [result.indices, result.summary, optuna_path]


def stage_evaluation(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """§11 evaluation suite: run every metric family, write reports/eval/metrics.json.

    Runs ``scripts/eval_emulator.py`` as a subprocess (it loads torch + the whole
    evaluation package); its metrics.json is what figures 2/5/7/12/13 read, so
    wiring it into the driver is what makes a clean `make smoke`/`make all`
    produce all 13 figures rather than 8.

    The script exits nonzero if any family raises (the right behavior for the
    standalone gate). Here, because it writes metrics.json before exiting, a
    partial run is DEGRADED, not FAILED: figures use the families that succeeded
    and the eval report records which failed. A run that never wrote metrics.json
    is a real failure and propagates.
    """
    from simworld.pipeline import Degraded

    metrics = Path(cfg.paths.reports) / "eval" / "metrics.json"
    try:
        _run_script(cfg, "eval_emulator.py", group="all")
    except subprocess.CalledProcessError as exc:
        if not metrics.is_file():
            raise
        raise Degraded(
            "evaluation suite: some metric families failed (see reports/eval/report.md)",
            outputs=[str(metrics)],
        ) from exc
    if metrics.is_file():
        tracker.log_artifact(metrics, "eval_metrics")
    return [metrics] if metrics.is_file() else []


def stage_figures(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 15: the 13 paper figures (Matplotlib) + Plotly exploration artifacts."""
    from simworld.visualization.figures import make_all_figures
    from simworld.visualization.interactive import make_all_interactive

    figures = make_all_figures(cfg)
    interactive = make_all_interactive(cfg)
    tracker.log_metrics({"figures_written": float(len(figures) + len(interactive))})
    return [*figures, *interactive]


def stage_report(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 17: assemble reports/FINDINGS.md — claims ledger + required failure section."""
    from simworld.evaluation.report import build_findings

    findings = build_findings(cfg)
    return [findings]
