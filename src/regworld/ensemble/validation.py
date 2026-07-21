"""§10 Stage 11 — ABM cross-validation subsample (the OOD-honesty check).

Re-run a stratified sample of the scenario cube's static-policy cells in the
TRUE tensorized ABM, driven by actual Stage-4 posterior theta draws, and check
how often the ABM's terminal outcome falls inside the emulator's own
predictive interval for that policy. That fraction is the coverage number: the
Phase-6 gate wants it >= 0.85 at ``dev``; at ``smoke`` it is reported without
gating (§10 Stage 11).

Learned policies (``rl_ppo``, ``rl_dreamer``) are out of scope for this
cross-check: they do not reduce to the single static ``PolicyLevers`` vector
that ``rollout_tensorized`` takes, so only the static-policy cells of the cube
are validated here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from regworld import rules
from regworld.abm.model import load_observed_world
from regworld.abm.policies import STATIC_POLICIES
from regworld.abm.tensorized import TensorTrajectory, rollout_tensorized
from regworld.models.world_model import WorldModel
from regworld.training.datamodule import load_theta_draws
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

# cube column name -> rules.QuarterOutcome field name
_METRIC_FIELD: dict[str, str] = {
    "compliance_rate": "compliance_rate",
    "hhi": "hhi",
    "mean_trust": "mean_trust",
    "consumer_surplus": "consumer_surplus",
    "exit_rate": "exit_rate_cum",
}
METRICS: tuple[str, ...] = tuple(_METRIC_FIELD)
COVERAGE_THRESHOLD_DEV = 0.85


@dataclass
class ValidationReport:
    coverage: float
    n_validated: int
    per_policy: list[dict[str, Any]]
    path: Path


def validation_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "ensemble"


def _terminal_tensor_outcome(trajectory: TensorTrajectory) -> rules.QuarterOutcome:
    o = trajectory.outcomes[-1]
    return rules.QuarterOutcome(
        compliance_rate=float(o.compliance_rate.item()),
        compliance_rate_weighted=float(o.compliance_rate_weighted.item()),
        compliance_by_tercile=tuple(float(x) for x in o.compliance_by_tercile.tolist()),  # type: ignore[arg-type]
        hhi=float(o.hhi.item()),
        mean_trust=float(o.mean_trust.item()),
        consumer_surplus=float(o.consumer_surplus.item()),
        exit_rate_cum=float(o.exit_rate_cum.item()),
        enforcement_cost=float(o.enforcement_cost.item()),
        n_audits=round(float(o.n_audits.item())),
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))


def _quantile(series: pl.Series, q: float) -> float:
    value = series.quantile(q, interpolation="linear")
    return float(value) if value is not None else float("nan")


def run_validation(
    cfg: RegWorldConfig, cube: pl.DataFrame, model: WorldModel, meta: dict[str, Any]
) -> ValidationReport:
    """Cross-validate a ``validation_frac`` subsample of the cube against the ABM."""
    del model, meta  # the ABM cross-check needs no emulator; signature kept stable
    path = validation_dir(cfg) / "validation_report.json"

    static_policies = (
        sorted({p for p in cube["policy"].unique().to_list() if p in STATIC_POLICIES})
        if cube.height
        else []
    )
    if not static_policies:
        report = {
            "coverage": float("nan"),
            "n_validated": 0,
            "per_policy": [],
            "note": "no static-policy cells in the cube to validate",
        }
        _write_report(path, report)
        return ValidationReport(float("nan"), 0, [], path)

    try:
        theta_rows = load_theta_draws(cfg)
    except FileNotFoundError as exc:
        report = {
            "coverage": float("nan"),
            "n_validated": 0,
            "per_policy": [],
            "note": f"no calibrated posterior available: {exc}",
        }
        _write_report(path, report)
        return ValidationReport(float("nan"), 0, [], path)

    theta_names = list(rules.Theta.__dataclass_fields__)
    world = load_observed_world(cfg, seed=cfg.seed)

    rng = np.random.default_rng(cfg.seed + 77_000)
    hits = 0
    total = 0
    per_policy: list[dict[str, Any]] = []
    for policy in static_policies:
        policy_cells = cube.filter(pl.col("policy") == policy)
        available_draws = policy_cells["draw"].unique().to_list()
        n_val = max(1, round(cfg.ensemble.validation_frac * max(len(available_draws), 1)))
        n_pick = min(n_val, len(available_draws))
        picked = rng.choice(np.asarray(available_draws), size=n_pick, replace=False)
        intervals = {
            metric: (
                _quantile(policy_cells[metric], 0.05),
                _quantile(policy_cells[metric], 0.95),
            )
            for metric in METRICS
        }
        policy_hits = 0
        policy_total = 0
        for draw in picked:
            draw_idx = int(draw)
            theta_row = theta_rows[draw_idx % theta_rows.shape[0]]
            theta = rules.Theta(**dict(zip(theta_names, theta_row.tolist(), strict=True)))
            seed = cfg.seed + 88_000 + draw_idx
            trajectory = rollout_tensorized(
                cfg,
                world,
                theta,
                STATIC_POLICIES[policy],
                seed=seed,
                quarters=cfg.horizon_quarters,
            )
            outcome = _terminal_tensor_outcome(trajectory)
            policy_total += 1
            for metric, field_name in _METRIC_FIELD.items():
                value = float(getattr(outcome, field_name))
                lo, hi = intervals[metric]
                covered = lo <= value <= hi
                hits += int(covered)
                total += 1
                policy_hits += int(covered)
        per_policy.append(
            {
                "policy": policy,
                "n_validated": policy_total,
                "intervals": intervals,
                "coverage": (
                    float(policy_hits / (len(METRICS) * policy_total)) if policy_total else None
                ),
            }
        )

    coverage = float(hits / total) if total else float("nan")
    report = {
        "coverage": coverage,
        "n_validated": total,
        "per_policy": per_policy,
        "threshold_dev": COVERAGE_THRESHOLD_DEV,
        "note": (
            "reported without gating at smoke"
            if cfg.profile_name == "smoke"
            else f"gate: coverage >= {COVERAGE_THRESHOLD_DEV}"
        ),
    }
    _write_report(path, report)
    return ValidationReport(coverage=coverage, n_validated=total, per_policy=per_policy, path=path)


__all__ = ["COVERAGE_THRESHOLD_DEV", "METRICS", "ValidationReport", "run_validation"]
