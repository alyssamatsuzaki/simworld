"""Stage implementations for the driver. Filled in phase by phase (§10).

Contract: `stage_<name>(cfg, tracker) -> list[Path]` of durable outputs.
Raise `regworld.pipeline.Degraded(note)` for an honest partial result.
Heavy imports stay inside the functions: the driver process must not pay for
(or conflict with) libraries a disabled stage would have used. Calibration is
always launched as a subprocess so JAX never enters this process (§5).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from regworld.tracking import Tracker
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def _run_script(
    cfg: RegWorldConfig,
    script: str,
    extra_overrides: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a stage script as a subprocess with the current profile (JAX isolation, §5)."""
    cmd = [sys.executable, f"scripts/{script}", f"profile={cfg.profile_name}"]
    cmd += extra_overrides or []
    full_env = dict(os.environ)
    full_env.setdefault("JAX_PLATFORMS", "cpu")
    full_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    full_env.update(env or {})
    log.info("subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=full_env)


def stage_recon(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
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


def stage_data(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 1: generate the world, degrade it, ingest the analysis panel (§10)."""
    from regworld.data.duck import build_views
    from regworld.data.generate import generate_ground_truth
    from regworld.data.ingest import ingest

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


def stage_graphs(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 2: observed edges -> metrics + PyG HeteroData (§10)."""
    import json

    import polars as pl
    import torch

    from regworld.data.store import observed_dir
    from regworld.graphs.to_pyg import hetero_from_edges, static_node_features

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


def stage_abm(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 3: observed-world Mesa model under the configured policy."""
    from regworld.abm.collect import run_observed_abm

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


def stage_tensorized_abm(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 3b: differentiable sparse-tensor simulation on observed inputs."""
    import polars as pl

    from regworld.abm.collect import run_tensorized_abm

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


def stage_calibration(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 4: launch JAX inference out of process and register its artifacts."""
    _run_script(cfg, "calibrate.py")
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
    return [manifest_path, *[Path(path) for path in manifest["outputs"]]]


def stage_causal(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 5: causal estimates, refutation, discovery, and the four-number gate."""
    from regworld.causal.gate import run_gate, write_gate_outputs
    from regworld.pipeline import Degraded

    _run_script(cfg, "causal_analysis.py")
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


def stage_emulator(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 5, Stages 6-7")


def stage_envs(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 8 checkpoint: validate and record the current Gymnasium contract."""
    from regworld.environments.abm_env import AbmEnv

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


def stage_marl(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Stage 9 checkpoint: exercise one strategic Parallel-API transition."""
    from regworld.environments.marl_env import RegulationMARLEnv

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


def stage_rl(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 10")


def stage_ensemble(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 11")


def stage_sensitivity(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 14")


def stage_figures(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 7, Stage 15")


def stage_report(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 7, Stage 17")
