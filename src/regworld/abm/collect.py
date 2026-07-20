"""Mesa DataCollector setup and durable ABM trajectory outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from mesa.datacollection import DataCollector

from regworld.abm.agents import FirmAgent
from regworld.types import RegWorldConfig

if TYPE_CHECKING:
    from regworld.abm.model import RegulationModel, Trajectory


def make_data_collector() -> DataCollector:
    """Collect aggregate outcomes plus firm state through Mesa's native collector."""
    return DataCollector(
        model_reporters={
            "quarter": "quarter",
            "compliance_rate": "compliance_rate",
            "compliance_rate_weighted": "compliance_rate_weighted",
            "compliance_small": "compliance_small",
            "compliance_mid": "compliance_mid",
            "compliance_large": "compliance_large",
            "hhi": "hhi",
            "mean_trust": "mean_trust",
            "consumer_surplus": "consumer_surplus",
            "exit_rate": "exit_rate",
            "enforcement_cost": "enforcement_cost",
            "n_audits": "n_audits",
            "reward": "reward",
            "backfire": "backfire",
        },
        agenttype_reporters={
            FirmAgent: {
                "firm_id": "firm_id",
                "quarter": "quarter",
                "compliant": "compliant",
                "alive": "alive",
                "revenue": "revenue",
                "audited": "audited",
                "fined": "fined",
                "profit_reward": "profit_reward",
            }
        },
    )


def model_frame_from_collector(model: RegulationModel) -> pl.DataFrame:
    pandas_frame = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    if pandas_frame.empty:
        return pl.DataFrame(
            schema={
                "quarter": pl.Int64,
                "compliance_rate": pl.Float64,
                "compliance_rate_weighted": pl.Float64,
                "compliance_small": pl.Float64,
                "compliance_mid": pl.Float64,
                "compliance_large": pl.Float64,
                "hhi": pl.Float64,
                "mean_trust": pl.Float64,
                "consumer_surplus": pl.Float64,
                "exit_rate": pl.Float64,
                "enforcement_cost": pl.Float64,
                "n_audits": pl.Int64,
                "reward": pl.Float64,
                "backfire": pl.Boolean,
            }
        )
    return pl.from_pandas(pandas_frame).cast(
        {
            "quarter": pl.Int64,
            "n_audits": pl.Int64,
            "backfire": pl.Boolean,
        }
    )


def firm_panel_from_collector(model: RegulationModel) -> pl.DataFrame:
    schema = pl.Schema(
        {
            "firm_id": pl.Int64,
            "quarter": pl.Int64,
            "compliant": pl.Boolean,
            "alive": pl.Boolean,
            "revenue": pl.Float64,
            "audited": pl.Boolean,
            "fined": pl.Boolean,
            "profit_reward": pl.Float64,
        }
    )
    if not model.cfg.abm.collect_agent_panel:
        return pl.DataFrame(schema=schema)
    pandas_frame = model.datacollector.get_agenttype_vars_dataframe(FirmAgent).reset_index()
    if pandas_frame.empty:
        return pl.DataFrame(schema=schema)
    frame = pl.from_pandas(pandas_frame)
    return frame.select(*schema).cast(schema)


def write_trajectory_outputs(
    cfg: RegWorldConfig,
    trajectory: Trajectory,
    *,
    seed: int,
    policy_name: str,
) -> list[Path]:
    out_dir = Path(cfg.paths.root) / "abm"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "trajectory.parquet"
    trajectory.aggregate.write_parquet(aggregate_path, compression="snappy")
    paths = [aggregate_path]
    if cfg.abm.collect_agent_panel:
        firm_path = out_dir / "firm_panel.parquet"
        trajectory.firm_panel.write_parquet(firm_path, compression="snappy")
        paths.append(firm_path)
    summary_path = out_dir / "summary.json"
    terminal = trajectory.outcomes[-1] if trajectory.outcomes else None
    summary = {
        "seed": seed,
        "policy": policy_name,
        "n_quarters": len(trajectory.outcomes),
        "n_firms": int(trajectory.final_state.y.size),
        "terminal": None
        if terminal is None
        else {
            "compliance_rate": terminal.compliance_rate,
            "hhi": terminal.hhi,
            "mean_trust": terminal.mean_trust,
            "exit_rate": terminal.exit_rate_cum,
        },
        "events": list(trajectory.events),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    paths.append(summary_path)
    return paths


def write_tensorized_outputs(
    cfg: RegWorldConfig,
    trajectory: Any,
    *,
    seed: int,
    policy_name: str,
) -> list[Path]:
    """Detach a tensor trajectory into portable aggregate Parquet and JSON."""
    matrix = trajectory.outcome_matrix().detach().cpu().numpy()
    aggregate = pl.DataFrame(
        {
            "quarter": range(1, matrix.shape[0] + 1),
            "compliance_rate": matrix[:, 0],
            "compliance_rate_weighted": matrix[:, 1],
            "hhi": matrix[:, 2],
            "mean_trust": matrix[:, 3],
            "consumer_surplus": matrix[:, 4],
            "exit_rate": matrix[:, 5],
            "enforcement_cost": matrix[:, 6],
        }
    )
    out_dir = Path(cfg.paths.root) / "abm"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "tensorized_trajectory.parquet"
    aggregate.write_parquet(aggregate_path, compression="snappy")
    summary_path = out_dir / "tensorized_summary.json"
    terminal = aggregate.tail(1).to_dicts()[0] if aggregate.height else None
    summary_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "policy": policy_name,
                "n_quarters": aggregate.height,
                "n_firms": int(trajectory.final_state.y.numel()),
                "terminal": terminal,
            },
            indent=2,
        )
    )
    return [aggregate_path, summary_path]


def run_observed_abm(
    cfg: RegWorldConfig,
    *,
    seed: int | None = None,
    include_tensorized: bool | None = None,
) -> tuple[Trajectory, list[Path]]:
    from regworld.abm.model import RegulationModel

    actual_seed = cfg.seed if seed is None else seed
    model = RegulationModel(cfg, seed=actual_seed)
    quarters = min(cfg.horizon_quarters, cfg.abm.max_quarters)
    trajectory = model.run(quarters)
    paths = write_trajectory_outputs(
        cfg,
        trajectory,
        seed=actual_seed,
        policy_name=cfg.policy.name,
    )
    should_tensorize = (
        cfg.stages.tensorized_abm if include_tensorized is None else include_tensorized
    )
    if should_tensorize:
        paths.extend(run_tensorized_abm(cfg, seed=actual_seed, world=model.world))
    return trajectory, paths


def run_tensorized_abm(
    cfg: RegWorldConfig,
    *,
    seed: int | None = None,
    world: Any | None = None,
) -> list[Path]:
    """Run Stage 3b independently so the pipeline can checkpoint it honestly."""
    from regworld.abm.model import load_observed_world
    from regworld.abm.policies import levers_from_config
    from regworld.abm.tensorized import rollout_tensorized

    actual_seed = cfg.seed if seed is None else seed
    observed_world = world or load_observed_world(cfg, actual_seed)
    quarters = min(cfg.horizon_quarters, cfg.abm.max_quarters)
    trajectory = rollout_tensorized(
        cfg,
        observed_world,
        observed_world.theta,
        levers_from_config(cfg.policy),
        actual_seed,
        quarters=quarters,
    )
    return write_tensorized_outputs(
        cfg,
        trajectory,
        seed=actual_seed,
        policy_name=cfg.policy.name,
    )
