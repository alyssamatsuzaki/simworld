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
    raise NotImplementedError("Phase 3, Stage 3")


def stage_tensorized_abm(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3, Stage 3b")


def stage_calibration(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 4, Stage 4")


def stage_causal(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 4, Stage 5")


def stage_emulator(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 5, Stages 6-7")


def stage_envs(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3/5, Stage 8")


def stage_marl(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3, Stage 9")


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
