"""Stage 1 acceptance tests (§10): contracts, determinism, observation sanity, firewall."""

from __future__ import annotations

import hashlib
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from regworld.data.generate import generate_ground_truth
from regworld.data.ingest import ingest, read_panel_analysis
from regworld.data.schema import ALL_OBSERVED, FIRM_PANEL, validate_table
from regworld.data.store import read_observed, read_oracle
from regworld.types import PopulationCfg, RegWorldConfig


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> RegWorldConfig:
    """A small generated world shared by every test in this module."""
    from regworld.types import validate_config

    from .conftest import compose_cfg

    tmp = tmp_path_factory.mktemp("world")
    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    cfg = cfg.model_copy(
        update={
            "population": PopulationCfg(
                name="small", n_firms=100, n_consumer_segments=6, n_regions=4
            )
        }
    )
    cfg.paths.root = str(tmp / "artifacts")
    cfg.paths.data = str(tmp / "artifacts/data")
    cfg.paths.graphs = str(tmp / "artifacts/graphs")
    cfg.paths.reports = str(tmp / "reports")
    generate_ground_truth(cfg)
    ingest(cfg)
    return cfg


def test_all_observed_tables_validate(world: RegWorldConfig) -> None:
    for name, spec in ALL_OBSERVED.items():
        validate_table(read_observed(world, name, validate=False), spec)


def test_row_counts_and_keys(world: RegWorldConfig) -> None:
    registry = read_observed(world, "firm_registry")
    assert registry.height == 100
    panel = read_observed(world, "firm_panel")
    n_sampled = panel["firm_id"].n_unique()
    assert n_sampled == max(round(world.dgp.panel_sample_frac * 100), 10)
    # balanced panel: every sampled firm appears in every observed quarter
    assert panel.height == n_sampled * world.observed_quarters
    agg = read_observed(world, "aggregate_series")
    assert agg.height == world.observed_quarters


def test_raw_panel_matches_plan_contract_exactly(world: RegWorldConfig) -> None:
    panel = read_observed(world, "firm_panel")
    assert panel.columns == list(FIRM_PANEL.schema)
    assert panel.schema["reported_compliant"] == pl.Boolean
    hidden_regressors = {
        "perceived_risk",
        "cost_share",
        "neighbor_compliant_share",
        "assoc_compliant_share",
        "privacy_rev_share",
        "phase_phi",
        "compliant_lag",
        "capacity_z",
    }
    assert hidden_regressors.isdisjoint(panel.columns)


def test_validation_rejects_extra_and_nonfinite_columns(world: RegWorldConfig) -> None:
    panel = read_observed(world, "firm_panel")
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_table(panel.with_columns(pl.lit(1).alias("hidden_truth")), FIRM_PANEL)
    for value in (float("nan"), float("inf"), float("-inf")):
        bad = pl.concat(
            [
                panel.head(1).with_columns(pl.lit(value, dtype=pl.Float64).alias("revenue_noisy")),
                panel.slice(1),
            ]
        )
        with pytest.raises(ValueError, match="non-finite"):
            validate_table(bad, FIRM_PANEL)


def test_registry_has_no_answer_key_columns(world: RegWorldConfig) -> None:
    registry = read_observed(world, "firm_registry")
    forbidden = {"capacity", "z", "capacity_z", "beta", "size"}  # continuous size is hidden too
    assert not forbidden & set(registry.columns)


def test_observation_model_sanity(world: RegWorldConfig) -> None:
    """Observed aggregates track the oracle truth within noise (3-4 sigma_obs)."""
    agg = read_observed(world, "aggregate_series").sort("quarter")
    truth = read_oracle(world, "regime_p_full")
    true_rates = (
        truth.filter(pl.col("quarter") <= world.observed_quarters)
        .group_by("quarter")
        .agg((pl.col("compliant") * pl.col("alive")).sum() / pl.col("alive").sum())
        .sort("quarter")
    )
    diff = np.abs(agg["compliance_rate_obs"].to_numpy() - true_rates.to_numpy()[:, 1].astype(float))
    assert diff.max() < 4 * world.dgp.sigma_obs + 1e-6


def test_compliance_actually_rises_in_regime_p(world: RegWorldConfig) -> None:
    """The world is not degenerate: enforcement produces adoption."""
    truth = read_oracle(world, "regime_p_full")
    q_last = truth.filter(pl.col("quarter") == world.horizon_quarters)
    q_first = truth.filter(pl.col("quarter") == 1)
    assert q_last["compliant"].mean() > q_first["compliant"].mean() + 0.1


def test_generation_deterministic(world: RegWorldConfig, tmp_path: Path) -> None:
    """Same seed → byte-identical Parquet (checksums compared across two runs)."""
    cfg2 = world.model_copy(deep=True)
    cfg2.paths.root = str(tmp_path / "artifacts")
    cfg2.paths.data = str(tmp_path / "artifacts/data")
    cfg2.paths.graphs = str(tmp_path / "artifacts/graphs")
    generate_ground_truth(cfg2)
    for name in ("firm_panel", "firm_registry", "aggregate_series"):
        a = Path(world.paths.data) / "observed" / f"{name}.parquet"
        b = Path(cfg2.paths.data) / "observed" / f"{name}.parquet"
        ha = hashlib.sha256(a.read_bytes()).hexdigest()
        hb = hashlib.sha256(b.read_bytes()).hexdigest()
        assert ha == hb, f"{name} not byte-identical across identical seeds"


def test_ingest_alignment(world: RegWorldConfig) -> None:
    df = read_panel_analysis(world)
    assert int(df["quarter"].max()) == world.observed_quarters - 1  # last decision quarter
    assert int(df["quarter"].min()) >= 1
    treated = df.filter(pl.col("treated") == 1.0)
    assert (treated["quarter"] >= treated["treatment_quarter"]).all()
    # event_time is quarter - treatment_quarter wherever treatment exists
    et = df.filter(pl.col("treatment_quarter") > 0)
    assert (et["event_time"] == et["quarter"] - et["treatment_quarter"]).all()


def test_ingest_hats_come_from_observed_inputs(world: RegWorldConfig) -> None:
    panel = read_observed(world, "firm_panel")
    analysis = read_panel_analysis(world).sort("firm_id", "quarter")

    same_quarter = panel.select(
        "firm_id",
        "quarter",
        pl.col("reported_compliant").cast(pl.Float64).alias("lag_expected"),
    )
    next_report = panel.select(
        "firm_id",
        (pl.col("quarter") - 1).alias("quarter"),
        pl.col("reported_compliant").cast(pl.Float64).alias("outcome_expected"),
    )
    previous_audit = panel.select(
        "firm_id",
        (pl.col("quarter") + 1).alias("quarter"),
        pl.col("audited").cast(pl.Float64).alias("audit_expected"),
    )
    aligned = (
        analysis.join(same_quarter, on=["firm_id", "quarter"])
        .join(next_report, on=["firm_id", "quarter"])
        .join(previous_audit, on=["firm_id", "quarter"], how="left")
        .with_columns(pl.col("audit_expected").fill_null(0.0))
    )
    assert (aligned["compliant_lag"] == aligned["lag_expected"]).all()
    assert (aligned["outcome_reported"] == aligned["outcome_expected"]).all()
    assert (aligned["audited_prev"] == aligned["audit_expected"]).all()

    phase_expected = np.where(
        analysis["treatment_quarter"].to_numpy() > 0,
        np.clip(
            (analysis["quarter"].to_numpy() - analysis["treatment_quarter"].to_numpy() + 1) / 9.0,
            0.0,
            1.0,
        ),
        0.0,
    )
    assert np.allclose(analysis["phase_phi"].to_numpy(), phase_expected)

    edges = pl.read_parquet(Path(world.paths.data) / "observed/graphs/supply_edges.parquet")
    neighbours: defaultdict[int, set[int]] = defaultdict(set)
    for src, dst in edges.iter_rows():
        i = int(str(src).removeprefix("firm_"))
        j = int(str(dst).removeprefix("firm_"))
        neighbours[i].add(j)
        neighbours[j].add(i)
    reports = {
        (int(row["firm_id"]), int(row["quarter"])): float(row["reported_compliant"])
        for row in panel.iter_rows(named=True)
    }
    alive = {
        (int(row["firm_id"]), int(row["quarter"])): bool(row["alive"])
        for row in panel.iter_rows(named=True)
    }
    expected_neighbour = []
    for row in analysis.iter_rows(named=True):
        firm_id, quarter = int(row["firm_id"]), int(row["quarter"])
        values = [
            reports[(other, quarter)]
            for other in neighbours[firm_id]
            if (other, quarter) in reports
            and (quarter == 1 or alive.get((other, quarter - 1), False))
        ]
        expected_neighbour.append(float(np.mean(values)) if values else 0.0)
    assert np.allclose(analysis["neighbor_compliant_share"], expected_neighbour)

    hats = [
        "perceived_risk",
        "cost_share",
        "neighbor_compliant_share",
        "assoc_compliant_share",
        "privacy_rev_share",
        "phase_phi",
        "compliant_lag",
        "log_size_proxy",
    ]
    for name in hats:
        values = analysis[name]
        assert values.is_finite().all(), name
        assert values.n_unique() > 1, name


def test_real_panel_path_adapter(world: RegWorldConfig, tmp_path: Path) -> None:
    cfg = world.model_copy(deep=True)
    cfg.paths.root = str(tmp_path / "artifacts")
    cfg.paths.data = str(tmp_path / "artifacts/data")
    observed = Path(cfg.paths.data) / "observed"
    shutil.copytree(Path(world.paths.data) / "observed", observed)
    real_panel = observed / "firm_panel.parquet"
    cfg.data = cfg.data.model_copy(update={"source": "real", "real_panel_path": str(real_panel)})
    ingest(cfg)
    assert_frame_equal(
        read_panel_analysis(cfg).sort("firm_id", "quarter"),
        read_panel_analysis(world).sort("firm_id", "quarter"),
    )


def test_historical_ingest_is_invariant_to_forecast_policy(world: RegWorldConfig) -> None:
    expected = read_panel_analysis(world).sort("firm_id", "quarter")
    cfg = world.model_copy(deep=True)
    cfg.policy = cfg.policy.model_copy(
        update={
            "enforcement": 0.0,
            "targeting": -1.0,
            "phase_speed": 1.0,
            "subsidy": 1.0,
        }
    )
    ingest(cfg)
    assert_frame_equal(
        read_panel_analysis(cfg).sort("firm_id", "quarter"),
        expected,
    )


def test_true_effects_exist_and_positive(world: RegWorldConfig) -> None:
    eff = read_oracle(world, "true_effects")
    assert eff["tau_true_onset_att"] > 0.0, "enforcement onset must raise compliance"
    assert eff["tau_true_audit_ate"] >= 0.0
    assert len(eff["tau_true_onset_per_quarter"]) == world.observed_quarters


def test_oracle_guard_blocks_unauthorized_caller(world: RegWorldConfig, tmp_path: Path) -> None:
    mod = tmp_path / "sneaky_module.py"
    mod.write_text(
        textwrap.dedent(
            """
            from regworld.data.store import read_oracle

            def peek(cfg):
                return read_oracle(cfg, "theta_star")
            """
        )
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("sneaky_module", mod)
    assert spec is not None and spec.loader is not None
    sneaky = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sneaky)
    with pytest.raises(RuntimeError, match="firewall"):
        sneaky.peek(world)


def test_oracle_guard_allows_tests(world: RegWorldConfig) -> None:
    theta = read_oracle(world, "theta_star")
    assert theta["beta_peer"] == 1.4
