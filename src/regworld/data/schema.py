"""Data contracts (§8): Polars schema dicts, validated on read and write.

A column-type surprise at quarter three of the project is a wasted week.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class TableSpec:
    name: str
    schema: dict[str, pl.DataType]
    non_null: tuple[str, ...] = ()
    unique_key: tuple[str, ...] = ()
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


F64 = pl.Float64()
I64 = pl.Int64()
BOOL = pl.Boolean()

FIRM_REGISTRY = TableSpec(
    "firm_registry",
    {
        "firm_id": I64,
        "sector": I64,
        "size_decile": I64,
        "data_intensity": F64,
        "association": I64,
        "cost_index": F64,
    },
    non_null=("firm_id", "sector", "size_decile"),
    unique_key=("firm_id",),
    ranges={"size_decile": (0, 9), "data_intensity": (0.0, 1.0)},
)

FIRM_PANEL = TableSpec(
    "firm_panel",
    {
        "firm_id": I64,
        "quarter": I64,
        "region": I64,
        "treatment_quarter": I64,
        "reported_compliant": F64,
        "revenue_noisy": F64,
        "audited": BOOL,
        "fined": BOOL,
        "alive": BOOL,
        "perceived_risk": F64,
        "cost_share": F64,
        "neighbor_compliant_share": F64,
        "assoc_compliant_share": F64,
        "privacy_rev_share": F64,
        "phase_phi": F64,
        "compliant_lag": F64,
    },
    non_null=("firm_id", "quarter", "region", "reported_compliant"),
    unique_key=("firm_id", "quarter"),
    ranges={
        "reported_compliant": (0.0, 1.0),
        "neighbor_compliant_share": (0.0, 1.0),
        "assoc_compliant_share": (0.0, 1.0),
        "phase_phi": (0.0, 1.0),
        "compliant_lag": (0.0, 1.0),
    },
)

AGGREGATE_SERIES = TableSpec(
    "aggregate_series",
    {
        "quarter": I64,
        "compliance_rate_obs": F64,
        "compliance_rate_weighted_obs": F64,
        "hhi_obs": F64,
        "mean_trust_obs": F64,
        "exit_rate_obs": F64,
    },
    non_null=("quarter",),
    unique_key=("quarter",),
)

CONSUMER_SURVEY = TableSpec(
    "consumer_survey",
    {"segment_id": I64, "quarter": I64, "trust_reported": F64, "privacy_bucket": I64},
    non_null=("segment_id", "quarter"),
    unique_key=("segment_id", "quarter"),
    ranges={"trust_reported": (0.0, 1.0), "privacy_bucket": (0, 2)},
)

MARKET = TableSpec(
    "market",
    {"quarter": I64, "sector": I64, "revenue_share_rounded": F64},
    non_null=("quarter", "sector"),
    unique_key=("quarter", "sector"),
    ranges={"revenue_share_rounded": (0.0, 1.0)},
)

EDGES = TableSpec(
    "edges",
    {"src": pl.Utf8(), "dst": pl.Utf8()},
    non_null=("src", "dst"),
)

PANEL_ANALYSIS = TableSpec(
    "panel_analysis",
    {
        "firm_id": I64,
        "quarter": I64,  # the DECISION quarter t; outcome is y_t reported at t+1
        "region": I64,
        "treatment_quarter": I64,
        "outcome_reported": F64,
        "treated": F64,
        "event_time": I64,
        "perceived_risk": F64,
        "cost_share": F64,
        "neighbor_compliant_share": F64,
        "assoc_compliant_share": F64,
        "privacy_rev_share": F64,
        "phase_phi": F64,
        "compliant_lag": F64,
        "audited_prev": F64,
        "log_size_proxy": F64,
        "sector": I64,
        "size_decile": I64,
        "data_intensity": F64,
        "cost_index": F64,
        "alive": BOOL,
    },
    non_null=("firm_id", "quarter", "outcome_reported"),
    unique_key=("firm_id", "quarter"),
)

ALL_OBSERVED = {
    s.name: s for s in (FIRM_REGISTRY, FIRM_PANEL, AGGREGATE_SERIES, CONSUMER_SURVEY, MARKET)
}


def validate_table(df: pl.DataFrame, spec: TableSpec) -> None:
    """Raise ValueError on any schema/nullability/range/key violation."""
    problems: list[str] = []
    for col, dtype in spec.schema.items():
        if col not in df.columns:
            problems.append(f"missing column {col}")
        elif df.schema[col] != dtype:
            problems.append(f"{col}: expected {dtype}, got {df.schema[col]}")
    for col in spec.non_null:
        if col in df.columns and df[col].null_count() > 0:
            problems.append(f"{col}: {df[col].null_count()} nulls")
    for col, (lo, hi) in spec.ranges.items():
        if col in df.columns and df[col].dtype in (pl.Float64, pl.Int64) and df.height > 0:
            mn, mx = float(df[col].min()), float(df[col].max())  # type: ignore[arg-type]
            if mn < lo - 1e-9 or mx > hi + 1e-9:
                problems.append(f"{col}: values [{mn}, {mx}] outside [{lo}, {hi}]")
    if spec.unique_key and df.height > 0:
        dup = df.height - df.select(spec.unique_key).unique().height
        if dup:
            problems.append(f"unique key {spec.unique_key} violated by {dup} rows")
    if problems:
        raise ValueError(f"table {spec.name!r} failed validation: " + "; ".join(problems))
