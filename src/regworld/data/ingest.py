"""Observed tables → decision-aligned analysis panel (§10 Stage 1).

The raw panel contains measurements only.  Every decision-time regressor below is
an analyst-side estimate built from those measurements, the degraded observed graph,
and the published rollout calendar; no simulator covariate enters this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from regworld.data import store
from regworld.data.schema import EDGES, FIRM_PANEL, PANEL_ANALYSIS, validate_table
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

# Regime P's public implementation schedule phases in over nine quarters.  This is
# program metadata, unlike the latent firm-level phase covariate formerly copied from
# the DGP.  Real-data users should adapt this constant to their published schedule.
_ROLLOUT_PHASE_QUARTERS = 9.0
_REGIME_P_SUBSIDY = 0.3


def _read_real_panel(cfg: RegWorldConfig) -> pl.DataFrame:
    raw_path = cfg.data.real_panel_path
    if not raw_path:
        raise ValueError("data.real_panel_path is required when data.source='real'")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"real panel not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        panel = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        panel = pl.read_csv(path)
    else:
        raise ValueError("data.real_panel_path must be a .parquet, .pq, or .csv file")
    validate_table(panel, FIRM_PANEL)
    return panel


def _read_observed_edges(cfg: RegWorldConfig, name: str) -> pl.DataFrame:
    path = store.observed_dir(cfg) / "graphs" / f"{name}.parquet"
    edges = pl.read_parquet(path)
    validate_table(edges, EDGES)
    return edges


def _read_supply_edges(cfg: RegWorldConfig) -> pl.DataFrame:
    edges = _read_observed_edges(cfg, "supply_edges")
    # Supply graphs currently serialize firm ids as "0"; accepting "firm_0" keeps
    # ingest compatible with the heterogeneous graph convention without changing it.
    return (
        edges.select(
            pl.col("src").str.replace(r"^firm_", "").cast(pl.Int64, strict=False),
            pl.col("dst").str.replace(r"^firm_", "").cast(pl.Int64, strict=False),
        )
        .drop_nulls()
        .filter(pl.col("src") != pl.col("dst"))
    )


def _observed_neighbor_shares(panel: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    prior_alive = panel.select(
        "firm_id", (pl.col("quarter") + 1).alias("quarter"), pl.col("alive").alias("alive_start")
    )
    reports = (
        panel.select(
            "firm_id",
            "quarter",
            pl.col("reported_compliant").cast(pl.Float64).alias("compliant_lag"),
        )
        .join(prior_alive, on=["firm_id", "quarter"], how="left")
        .with_columns(pl.col("alive_start").fill_null(True))
    )
    neighbours = pl.concat(
        [
            edges.select(pl.col("src").alias("firm_id"), pl.col("dst").alias("neighbor_id")),
            edges.select(pl.col("dst").alias("firm_id"), pl.col("src").alias("neighbor_id")),
        ]
    ).unique()
    neighbour_reports = reports.select(
        pl.col("firm_id").alias("neighbor_id"), "quarter", "compliant_lag", "alive_start"
    )
    return (
        neighbours.join(neighbour_reports, on="neighbor_id", how="inner")
        .filter(pl.col("alive_start"))
        .group_by("firm_id", "quarter")
        .agg(pl.col("compliant_lag").mean().alias("neighbor_compliant_share"))
    )


def _association_shares(
    panel: pl.DataFrame, registry: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    prior_alive = panel.select(
        "firm_id", (pl.col("quarter") + 1).alias("quarter"), pl.col("alive").alias("alive_start")
    )
    reports = (
        panel.select(
            "firm_id",
            "quarter",
            pl.col("reported_compliant").cast(pl.Float64).alias("compliant_lag"),
        )
        .join(prior_alive, on=["firm_id", "quarter"], how="left")
        .join(registry.select("firm_id", "association"), on="firm_id", how="left")
        .with_columns(pl.col("alive_start").fill_null(True))
        .filter(pl.col("alive_start"))
    )
    overall = reports.group_by("quarter").agg(
        pl.col("compliant_lag").mean().alias("compliance_overall")
    )
    by_association = (
        reports.filter(pl.col("association") >= 0)
        .group_by("association", "quarter")
        .agg(pl.col("compliant_lag").mean().alias("compliance_association"))
    )
    return by_association, overall


def _privacy_market_proxy(
    survey: pl.DataFrame, market_edges: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Market links can be serialized in either direction. Quarter t uses survey t-1,
    # the latest release available when the firm makes its decision.
    links = (
        market_edges.select(
            pl.when(pl.col("src").str.starts_with("seg_"))
            .then(pl.col("src"))
            .otherwise(pl.col("dst"))
            .str.replace(r"^seg_", "")
            .cast(pl.Int64, strict=False)
            .alias("segment_id"),
            pl.when(pl.col("src").str.starts_with("firm_"))
            .then(pl.col("src"))
            .otherwise(pl.col("dst"))
            .str.replace(r"^firm_", "")
            .cast(pl.Int64, strict=False)
            .alias("firm_id"),
        )
        .drop_nulls()
        .unique()
    )
    privacy_prev = survey.select(
        "segment_id",
        (pl.col("quarter") + 1).alias("quarter"),
        (pl.col("privacy_bucket").cast(pl.Float64) / 2.0).alias("privacy_signal"),
    )
    linked = (
        links.join(privacy_prev, on="segment_id", how="inner")
        .group_by("firm_id", "quarter")
        .agg(pl.col("privacy_signal").mean().alias("privacy_linked"))
    )
    overall = privacy_prev.group_by("quarter").agg(
        pl.col("privacy_signal").mean().alias("privacy_overall")
    )
    return linked, overall


def ingest(cfg: RegWorldConfig) -> Path:
    panel = (
        _read_real_panel(cfg)
        if cfg.data.source == "real"
        else store.read_observed(cfg, "firm_panel")
    )
    registry = store.read_observed(cfg, "firm_registry")
    survey = store.read_observed(cfg, "consumer_survey")
    supply_edges = _read_supply_edges(cfg)
    market_edges = _read_observed_edges(cfg, "market_edges")

    size_proxy = registry.with_columns(
        (pl.col("size_decile").cast(pl.Float64) + 1.0).log().alias("log_size_raw")
    ).with_columns(
        (pl.col("log_size_raw") - pl.col("log_size_raw").median()).alias("log_size_proxy")
    )
    outcome_next = panel.select(
        "firm_id",
        (pl.col("quarter") - 1).alias("quarter"),
        pl.col("reported_compliant").cast(pl.Float64).alias("outcome_reported"),
    )
    audited_prev = panel.select(
        "firm_id",
        (pl.col("quarter") + 1).alias("quarter"),
        pl.col("audited").cast(pl.Float64).alias("audited_prev"),
    )
    regional_signals = (
        panel.group_by("region", "quarter")
        .agg(pl.col("audited").cast(pl.Float64).mean().alias("region_audit_prev"))
        .with_columns((pl.col("quarter") + 1).alias("quarter"))
    )
    sector_publicity = (
        panel.join(registry.select("firm_id", "sector"), on="firm_id", how="left")
        .group_by("sector", "quarter")
        .agg(pl.col("fined").cast(pl.Float64).mean().alias("sector_publicity_prev"))
        .with_columns((pl.col("quarter") + 1).alias("quarter"))
    )
    neighbour_shares = _observed_neighbor_shares(panel, supply_edges)
    association_shares, overall_shares = _association_shares(panel, registry)
    linked_privacy, overall_privacy = _privacy_market_proxy(survey, market_edges)

    joined = (
        panel.join(outcome_next, on=["firm_id", "quarter"], how="inner")
        .join(audited_prev, on=["firm_id", "quarter"], how="left")
        .join(regional_signals, on=["region", "quarter"], how="left")
        .join(size_proxy, on="firm_id", how="left")
        .join(sector_publicity, on=["sector", "quarter"], how="left")
        .join(neighbour_shares, on=["firm_id", "quarter"], how="left")
        .join(association_shares, on=["association", "quarter"], how="left")
        .join(overall_shares, on="quarter", how="left")
        .join(linked_privacy, on=["firm_id", "quarter"], how="left")
        .join(overall_privacy, on="quarter", how="left")
        .with_columns(
            pl.col("reported_compliant").cast(pl.Float64).alias("compliant_lag"),
            pl.col("audited_prev").fill_null(0.0),
            pl.col("region_audit_prev").fill_null(0.0),
            pl.col("sector_publicity_prev").fill_null(0.0),
            pl.col("neighbor_compliant_share").fill_null(0.0),
            pl.col("privacy_linked")
            .fill_null(pl.col("privacy_overall"))
            .fill_null(0.0)
            .alias("privacy_rev_share"),
            pl.when(pl.col("association") >= 0)
            .then(pl.col("compliance_association"))
            .otherwise(pl.col("compliance_overall"))
            .fill_null(pl.col("compliance_overall"))
            .fill_null(0.0)
            .alias("assoc_compliant_share"),
            pl.when(pl.col("treatment_quarter") > 0)
            .then((pl.col("quarter") >= pl.col("treatment_quarter")).cast(pl.Float64))
            .otherwise(0.0)
            .alias("treated"),
            pl.when(pl.col("treatment_quarter") > 0)
            .then(pl.col("quarter") - pl.col("treatment_quarter"))
            .otherwise(-999)
            .cast(pl.Int64)
            .alias("event_time"),
            pl.when(pl.col("treatment_quarter") > 0)
            .then(
                (
                    (pl.col("quarter") - pl.col("treatment_quarter") + 1).cast(pl.Float64)
                    / _ROLLOUT_PHASE_QUARTERS
                ).clip(0.0, 1.0)
            )
            .otherwise(0.0)
            .alias("phase_phi"),
        )
        .with_columns(
            (
                pl.col("region_audit_prev")
                * 4.0
                * (0.3 * pl.col("log_size_proxy")).exp()
                * pl.col("phase_phi")
                * (1.0 + 0.5 * pl.col("sector_publicity_prev"))
                * (1.0 + 0.8 * pl.col("audited_prev"))
            ).alias("perceived_risk"),
            (
                0.15
                * pl.col("cost_index").clip(0.0, 1_000_000.0)
                * pl.col("data_intensity")
                * (-0.5 * pl.col("log_size_proxy")).exp()
                * pl.col("phase_phi")
                * (1.0 - _REGIME_P_SUBSIDY * (pl.col("size_decile") <= 2).cast(pl.Float64))
            )
            .clip(0.0, 1_000_000.0)
            .alias("cost_share"),
        )
        .filter(pl.col("alive"))
        .select(
            "firm_id",
            "quarter",
            "region",
            "treatment_quarter",
            "outcome_reported",
            "treated",
            "event_time",
            "perceived_risk",
            "cost_share",
            "neighbor_compliant_share",
            "assoc_compliant_share",
            "privacy_rev_share",
            "phase_phi",
            "compliant_lag",
            "audited_prev",
            "log_size_proxy",
            "sector",
            "size_decile",
            "data_intensity",
            "cost_index",
            "alive",
        )
    )
    validate_table(joined, PANEL_ANALYSIS)
    if joined.is_empty():
        raise ValueError("analysis panel is empty after alignment and alive filtering")
    out = Path(cfg.paths.data) / "panel_analysis.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(out, compression="snappy")
    log.info(
        "analysis panel: %d rows, %d firms, quarters %d-%d",
        joined.height,
        joined["firm_id"].n_unique(),
        cast(int, joined["quarter"].min()),
        cast(int, joined["quarter"].max()),
    )
    return out


def read_panel_analysis(cfg: RegWorldConfig) -> pl.DataFrame:
    df = pl.read_parquet(Path(cfg.paths.data) / "panel_analysis.parquet")
    validate_table(df, PANEL_ANALYSIS)
    return df


def observation_sanity(cfg: RegWorldConfig) -> dict[str, float]:
    """Observed aggregate bounds used by the Stage 1 tests."""
    agg = store.read_observed(cfg, "aggregate_series")
    return {
        "mean_compliance_obs": cast(float, agg["compliance_rate_obs"].mean()),
        "sigma_obs": cfg.dgp.sigma_obs,
        "n_quarters": float(agg.height),
        "max_abs_obs": float(np.abs(agg["compliance_rate_obs"].to_numpy()).max()),
    }
