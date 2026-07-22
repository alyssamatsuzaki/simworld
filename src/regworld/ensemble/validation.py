"""§10 Stage 11 — ABM cross-validation subsample (the OOD-honesty check).

Re-run a stratified sample of the scenario cube's static-policy cells in the
TRUE tensorized ABM, driven by actual Stage-4 posterior theta draws, and check
how often the ABM's terminal outcome falls inside the emulator's own predictive
band for that policy. That fraction is the coverage number: the Phase-6 gate
wants it >= 0.85 at ``dev``/``full``; at ``smoke`` it is reported without gating
(§10 Stage 11) because 8 draws x 1 seed cannot estimate a 5%/95% band.

**What the interval actually is (read this before quoting the number).**
PLAN.md §10 Stage 11 asks for "the emulator's 90% predictive interval" per
(scenario, policy) cell. The trained GraphRSSM is *theta-marginal*: theta is
integrated out at training time by domain randomization (§10 Stages 6+7) and is
not an input to :class:`~regworld.models.world_model.WorldModel` or to
:class:`~regworld.environments.emulator_env.EmulatorEnv` — the only thing that
distinguishes two cube cells is the ``torch.Generator`` seed
(``ensemble.cube._cell_seed``). The cube's ``draw`` column is therefore a
nuisance index over that seed, *not* an identifier of a posterior draw, and no
per-theta-draw emulator band exists to be reconstructed from the cube.

So what is computed here is an honest **marginal** predictive-band coverage:

* emulator side — the 5%/95% quantiles of the terminal metric over *all* of a
  policy's cube cells, i.e. the emulator's terminal predictive distribution with
  theta already marginalized out;
* ABM side — the tensorized ABM's terminal outcome at a theta drawn uniformly
  from the Stage-4 posterior (so the ABM side is marginalized over the same
  posterior), under the same static policy.

Both sides are theta-marginal, so the comparison is internally matched; it is
*not* the conditional, per-draw check PLAN.md's wording implies. See
``INTERVAL_KIND`` and the ``interval_kind`` field of the written report — never
report this as a per-draw predictive interval.

Learned policies (``rl_ppo``, ``rl_dreamer``) are out of scope for this
cross-check: they do not reduce to the single static ``PolicyLevers`` vector
that ``rollout_tensorized`` takes, so only the static-policy cells of the cube
are validated here.
"""

from __future__ import annotations

import json
import logging
import math
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

#: §10 Stage 11 / §18: coverage must be >= this at any gating profile.
COVERAGE_THRESHOLD_DEV = 0.85

#: Profiles at which the threshold above is *not* enforced (too few draws).
UNGATED_PROFILES: frozenset[str] = frozenset({"smoke"})

#: Exactly what the reported coverage measures. Not a per-draw interval.
INTERVAL_KIND = (
    "emulator terminal 5%-95% band pooled over a policy's cube cells "
    "(theta-marginal); ABM side rolled out at theta ~ Stage-4 posterior"
)

# Gate outcomes.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_UNGATED = "UNGATED"  # reported, not enforced (smoke)
GATE_INDETERMINATE = "INDETERMINATE"  # nothing could be validated at a gating profile


class CoverageGateFailure(RuntimeError):
    """Raised when the Stage-11 coverage gate is enforced and does not pass.

    A plain ``RuntimeError`` subclass on purpose: ``pipeline.run_pipeline``'s
    generic handler records the stage FAILED (not DEGRADED), which is the honest
    status for a violated acceptance gate.
    """

    def __init__(
        self,
        note: str,
        *,
        coverage: float,
        threshold: float,
        status: str,
        path: Path | None = None,
    ) -> None:
        super().__init__(note)
        self.coverage = coverage
        self.threshold = threshold
        self.status = status
        self.path = path


def gate_is_enforced(cfg: RegWorldConfig) -> bool:
    """Whether the coverage threshold gates the run at this profile."""
    return cfg.profile_name not in UNGATED_PROFILES


def coverage_gate_status(cfg: RegWorldConfig, coverage: float) -> str:
    """Classify a coverage number against the §18 threshold for this profile.

    ``coverage`` is NaN exactly when nothing could be validated (no posterior,
    no static-policy cells): at a gating profile that is INDETERMINATE, which is
    not a pass — an un-computable gate has not been met.
    """
    if not gate_is_enforced(cfg):
        return GATE_UNGATED
    if not math.isfinite(coverage):
        return GATE_INDETERMINATE
    return GATE_PASS if coverage >= COVERAGE_THRESHOLD_DEV else GATE_FAIL


def _gate_failure(status: str, coverage: float, path: Path | None) -> CoverageGateFailure:
    where = f" (see {path})" if path is not None else ""
    if status == GATE_INDETERMINATE:
        note = (
            "Stage-11 coverage gate INDETERMINATE: nothing could be cross-validated "
            f"against the ABM{where}; the >= {COVERAGE_THRESHOLD_DEV} gate is unmet, "
            "not passed"
        )
    else:
        note = (
            f"Stage-11 coverage gate FAILED: coverage {coverage:.4f} < "
            f"{COVERAGE_THRESHOLD_DEV}{where} — the emulator's predictive band does not "
            "cover the ABM often enough; the ensemble is decoration until this is fixed"
        )
    return CoverageGateFailure(
        note,
        coverage=coverage,
        threshold=COVERAGE_THRESHOLD_DEV,
        status=status,
        path=path,
    )


def enforce_coverage_gate(
    cfg: RegWorldConfig, coverage: float, *, report_path: Path | None = None
) -> str:
    """Raise :class:`CoverageGateFailure` unless the coverage gate passes.

    Returns the gate status when it does not raise, so a caller can log it.
    Never raises at an ungated profile (``smoke``).
    """
    status = coverage_gate_status(cfg, coverage)
    if status in (GATE_PASS, GATE_UNGATED):
        return status
    raise _gate_failure(status, coverage, report_path)


@dataclass
class ValidationReport:
    coverage: float
    n_validated: int
    per_policy: list[dict[str, Any]]
    path: Path
    status: str = GATE_UNGATED
    gated: bool = False
    threshold: float = COVERAGE_THRESHOLD_DEV
    interval_kind: str = INTERVAL_KIND

    @property
    def passed(self) -> bool:
        """False only when the gate is enforced *and* not met."""
        return self.status in (GATE_PASS, GATE_UNGATED)

    def raise_for_gate(self) -> None:
        """Object-flavoured :func:`enforce_coverage_gate`."""
        if not self.passed:
            raise _gate_failure(self.status, self.coverage, self.path)


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


def _degraded_report(cfg: RegWorldConfig, path: Path, note: str) -> ValidationReport:
    """Write an honest NaN-coverage report; never crash the ensemble stage."""
    status = coverage_gate_status(cfg, float("nan"))
    gated = gate_is_enforced(cfg)
    _write_report(
        path,
        {
            "coverage": float("nan"),
            "n_validated": 0,
            "per_policy": [],
            "metric": "marginal_interval_coverage",
            "interval_kind": INTERVAL_KIND,
            "threshold": COVERAGE_THRESHOLD_DEV,
            "gated": gated,
            "status": status,
            "note": note,
        },
    )
    if gated:
        log.error("Stage-11 coverage gate %s: %s", status, note)
    else:
        log.warning("Stage-11 coverage not computed (%s): %s", status, note)
    return ValidationReport(
        coverage=float("nan"),
        n_validated=0,
        per_policy=[],
        path=path,
        status=status,
        gated=gated,
    )


def run_validation(
    cfg: RegWorldConfig, cube: pl.DataFrame, model: WorldModel, meta: dict[str, Any]
) -> ValidationReport:
    """Cross-validate a ``validation_frac`` subsample of the cube against the ABM.

    Never raises on the gate itself — the caller writes its artifacts first and
    then calls :func:`enforce_coverage_gate` (or
    :meth:`ValidationReport.raise_for_gate`). The returned report carries
    ``status`` / ``passed`` so nothing has to re-derive the verdict.
    """
    del model, meta  # the ABM cross-check needs no emulator; signature kept stable
    path = validation_dir(cfg) / "validation_report.json"

    static_policies = (
        sorted({p for p in cube["policy"].unique().to_list() if p in STATIC_POLICIES})
        if cube.height
        else []
    )
    if not static_policies:
        return _degraded_report(cfg, path, "no static-policy cells in the cube to validate")

    try:
        theta_rows = load_theta_draws(cfg)
    except FileNotFoundError as exc:
        return _degraded_report(cfg, path, f"no calibrated posterior available: {exc}")

    theta_names = list(rules.Theta.__dataclass_fields__)
    world = load_observed_world(cfg, seed=cfg.seed)

    rng = np.random.default_rng(cfg.seed + 77_000)

    # The cube's `draw` column indexes a torch.Generator seed, not a posterior
    # draw (see the module docstring), so pairing cube draw d with theta row d
    # would (a) mean nothing and (b) at smoke only ever reach the first 8 rows of
    # the posterior. Draw theta uniformly from the posterior instead, once per
    # cube draw index, so every policy is validated at the same theta and the
    # same seed — common random numbers across the policy comparison.
    all_draws = sorted({int(d) for d in cube["draw"].to_list()})
    theta_choice = rng.integers(theta_rows.shape[0], size=max(len(all_draws), 1))
    theta_index: dict[int, int] = {draw: int(theta_choice[i]) for i, draw in enumerate(all_draws)}

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
        metric_hits = dict.fromkeys(METRICS, 0)
        policy_hits = 0
        policy_total = 0
        theta_used: list[int] = []
        for draw in picked:
            draw_idx = int(draw)
            theta_idx = theta_index.get(draw_idx, 0)
            theta_used.append(theta_idx)
            theta_row = theta_rows[theta_idx]
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
                metric_hits[metric] += int(covered)
        per_policy.append(
            {
                "policy": policy,
                "n_validated": policy_total,
                "n_comparisons": policy_total * len(METRICS),
                "emulator_marginal_interval_05_95": intervals,
                "theta_rows_used": theta_used,
                "coverage": (
                    float(policy_hits / (len(METRICS) * policy_total)) if policy_total else None
                ),
                "per_metric_coverage": {
                    metric: (float(metric_hits[metric] / policy_total) if policy_total else None)
                    for metric in METRICS
                },
            }
        )

    coverage = float(hits / total) if total else float("nan")
    status = coverage_gate_status(cfg, coverage)
    gated = gate_is_enforced(cfg)
    report = {
        "coverage": coverage,
        # `n_validated` counts (cell, metric) comparisons, as it always has;
        # per_policy[*].n_validated counts ABM rollouts.
        "n_validated": total,
        "per_policy": per_policy,
        "metric": "marginal_interval_coverage",
        "interval_kind": INTERVAL_KIND,
        "threshold": COVERAGE_THRESHOLD_DEV,
        "gated": gated,
        "status": status,
        "note": (
            f"reported without gating at {cfg.profile_name}"
            if not gated
            else f"gate: coverage >= {COVERAGE_THRESHOLD_DEV} -> {status}"
        ),
    }
    _write_report(path, report)
    if status == GATE_FAIL:
        log.error(
            "Stage-11 coverage gate FAILED: %.4f < %.2f over %d comparisons (%s)",
            coverage,
            COVERAGE_THRESHOLD_DEV,
            total,
            path,
        )
    else:
        log.info(
            "Stage-11 coverage %.4f over %d comparisons (%s, threshold %.2f)",
            coverage,
            total,
            status,
            COVERAGE_THRESHOLD_DEV,
        )
    return ValidationReport(
        coverage=coverage,
        n_validated=total,
        per_policy=per_policy,
        path=path,
        status=status,
        gated=gated,
    )


__all__ = [
    "COVERAGE_THRESHOLD_DEV",
    "GATE_FAIL",
    "GATE_INDETERMINATE",
    "GATE_PASS",
    "GATE_UNGATED",
    "INTERVAL_KIND",
    "METRICS",
    "UNGATED_PROFILES",
    "CoverageGateFailure",
    "ValidationReport",
    "coverage_gate_status",
    "enforce_coverage_gate",
    "gate_is_enforced",
    "run_validation",
    "validation_dir",
]
