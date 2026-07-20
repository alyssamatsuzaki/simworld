"""C2 (§1, §10 Stage 5): the DiD recovers the truth; the careless numbers do not.

These are estimator-validation tests in the parameter-recovery spirit: they run on a
purpose-generated world with a *full* firm panel, where the estimators have power.
The shipped smoke world keeps its realistic 20% panel; its wide DiD interval is the
honest report of that sparsity, exercised by the Stage-5f gate instead.

The graded target for the DiD is the sealed ``tau_did_truth`` — the same group-time
estimator applied to the true panel — not ``tau_true_onset_att``: the DGP has
cross-region interference (peer and macro-trust spillovers reach not-yet-treated
controls), so the two estimands genuinely differ (DEVIATIONS 2026-07-20).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from regworld.causal.did import estimate_did
from regworld.causal.estimate import dml_audit, dml_onset
from regworld.causal.ground_truth import load_ground_truth
from regworld.causal.refute import refute_audit
from regworld.data.ingest import ingest, read_panel_analysis
from regworld.data.store import read_oracle
from regworld.types import RegWorldConfig

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dense_world(tmp_path_factory: pytest.TempPathFactory) -> RegWorldConfig:
    """A smoke-sized world observed with a full panel: estimators have power here."""
    from regworld.data.generate import generate_ground_truth
    from regworld.types import validate_config

    from .conftest import compose_cfg

    tmp = tmp_path_factory.mktemp("dense_world")
    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    cfg = cfg.model_copy(update={"dgp": cfg.dgp.model_copy(update={"panel_sample_frac": 1.0})})
    cfg.paths.root = str(tmp / "artifacts")
    cfg.paths.data = str(tmp / "artifacts/data")
    cfg.paths.graphs = str(tmp / "artifacts/graphs")
    cfg.paths.reports = str(tmp / "reports")
    generate_ground_truth(cfg)
    ingest(cfg)
    return cfg


def _report_scale(cfg: RegWorldConfig) -> float:
    """Misclassification shrinks reported effects by (1 - q0 - q1)."""
    return 1.0 - 2.0 * cfg.dgp.misclassification


def test_did_recovers_the_did_estimand(dense_world: RegWorldConfig) -> None:
    truth = load_ground_truth(dense_world)
    panel = read_panel_analysis(dense_world)
    did = estimate_did(panel, seed=0)
    scale = _report_scale(dense_world)
    ci_low, ci_high = did.ci_low / scale, did.ci_high / scale
    assert ci_low <= truth.did_truth <= ci_high, (
        f"de-attenuated DiD CI [{ci_low:.3f}, {ci_high:.3f}] "
        f"misses sealed tau_did_truth {truth.did_truth:.3f}"
    )
    # Pre-trends are flat by construction: well below the effect being measured.
    assert did.pretrend_max_abs < 0.5 * truth.did_truth


def test_interference_gap_is_real_and_positive(dense_world: RegWorldConfig) -> None:
    """Spillovers reach the controls, so the DiD estimand < the do() truth."""
    truth = load_ground_truth(dense_world)
    assert truth.interference_gap > 0.0
    assert truth.did_truth < truth.onset_att


def test_dml_is_precisely_wrong_for_the_policy_question(dense_world: RegWorldConfig) -> None:
    """The guide's warning made flesh: a tight CI around the wrong number."""
    truth = load_ground_truth(dense_world)
    panel = read_panel_analysis(dense_world)
    dml = dml_onset(panel, seed=0)
    bias = abs(dml.estimate - truth.onset_att)
    assert bias > 2.0 * dml.se, (
        f"the demonstration stopped demonstrating: DML bias {bias:.3f} "
        f"is within 2 SE ({dml.se:.3f}) of tau_true"
    )
    assert not (dml.ci_low <= truth.onset_att <= dml.ci_high)


def test_audit_effect_confounding_by_targeting(dense_world: RegWorldConfig) -> None:
    """Naive audit contrasts are confounded upward; observed controls cannot fix it."""
    truth = load_ground_truth(dense_world)
    panel = read_panel_analysis(dense_world)
    grouped = (
        panel.group_by((pl.col("audited_prev") > 0.5).alias("audited"))
        .agg(pl.col("outcome_reported").mean())
        .sort("audited")
    )
    rates = dict(
        zip(grouped["audited"].to_list(), grouped["outcome_reported"].to_list(), strict=True)
    )
    raw_contrast = float(rates[True] - rates[False])
    # Size targeting + capacity + enforcement timing all load onto the raw contrast.
    assert raw_contrast > 2.0 * truth.audit_ate
    dml = dml_audit(panel, seed=0)
    assert dml.estimate - truth.audit_ate > 2.0 * dml.se, (
        "DML with every observed control should still overstate the audit effect "
        f"(estimate {dml.estimate:.3f}, truth {truth.audit_ate:.3f}, se {dml.se:.3f})"
    )


def test_conditioning_on_sealed_capacity_documents_residual_dynamics(
    dense_world: RegWorldConfig,
) -> None:
    """Even do()-grade z does not close the audit gap: the residual is dynamic.

    This pins the DEVIATIONS finding: the planted static confounder is not the
    dominant bias source for the audit DML at these settings — enforcement-timing
    feedback is — so a would-be analyst cannot fix the estimate by measuring z.
    """
    truth = load_ground_truth(dense_world)
    panel = read_panel_analysis(dense_world)
    confounders = read_oracle(dense_world, "firm_confounders")
    z_by_firm = dict(
        zip(
            confounders["firm_id"].to_list(),
            confounders["capacity_z"].to_list(),
            strict=True,
        )
    )
    z = np.asarray([z_by_firm[int(f)] for f in panel["firm_id"].to_list()])
    with_z = dml_audit(panel, seed=0, extra_control=z)
    assert with_z.estimate - truth.audit_ate > 2.0 * with_z.se


def test_placebo_refuter_returns_zero(dense_world: RegWorldConfig) -> None:
    panel = read_panel_analysis(dense_world)
    report = refute_audit(panel, seed=0)
    assert abs(report.placebo_effect) < 0.05
    # Stability refuters should not move the estimate materially.
    assert abs(report.subset_effect - report.estimate) < 0.05
    assert report.e_value > 1.0


def test_discovery_fails_for_the_documented_reason(dense_world: RegWorldConfig) -> None:
    from regworld.causal.discovery import discover

    panel = read_panel_analysis(dense_world)
    report = discover(panel, seed=0)
    assert report.pc_shd > 0 and report.ges_shd > 0
    assert "latent capacity" in report.reason
