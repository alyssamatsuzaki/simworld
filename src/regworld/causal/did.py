"""Staggered-rollout difference-in-differences on the historical panel (§7.8, 5c-3).

Rollout timing is exogenous by construction, so a group-time ATT with not-yet-treated
and never-treated firms as controls identifies the onset ATT. Plain two-way
fixed-effects is *not* used: with staggered adoption and a dynamic (phase-in) effect
it makes forbidden already-treated-vs-newly-treated comparisons and can flip sign.
This is the Callaway-Sant'Anna clean-comparison estimator, aggregated over post cells
to match the answer key's estimand (mean over observed post-treatment cells), with a
firm-cluster bootstrap CI.

This is ``tau_qe`` in the four-number gate: what the data says under a credible
identification argument. The event study exposes the pre-trends - flat by
construction, and a bug in the DGP if they are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class DidResult:
    """Onset ATT with a 95% bootstrap CI, plus the event study for pre-trends."""

    att: float
    se: float
    ci_low: float
    ci_high: float
    event_times: list[int]
    event_coefficients: list[float]
    pretrend_max_abs: float
    n_cells: int
    n_obs: int

    def covers(self, value: float) -> bool:
        """True if ``value`` lies inside the 95% confidence interval."""
        return self.ci_low <= value <= self.ci_high


def group_time_att(
    y: np.ndarray,
    quarter: np.ndarray,
    treatment_quarter: np.ndarray,
) -> tuple[float, int, list[tuple[int, int, float, int]]]:
    """Aggregate ATT over (cohort g, period t) cells using not-yet-treated controls.

    Returns (weighted ATT, n_cells, per-cell records) where each record is
    ``(g, t, att_gt, n_treated_obs)``. Cells are weighted by treated-firm count to
    match the answer key's cell-averaged onset ATT.
    """
    treated_mask = treatment_quarter > 0
    cohorts = sorted({int(g) for g in treatment_quarter[treated_mask]})
    quarters = sorted({int(q) for q in quarter})
    records: list[tuple[int, int, float, int]] = []
    total_weight = 0.0
    weighted_sum = 0.0
    for g in cohorts:
        base = g - 1
        if base not in quarters:
            continue
        treated_g = treatment_quarter == g
        for t in quarters:
            if t < g:
                continue  # only post periods
            # not-yet-treated at t: never-treated, or treated strictly after t
            control = (treatment_quarter <= 0) | (treatment_quarter > t)
            for period in (t, base):
                if not (treated_g & (quarter == period)).any():
                    break
                if not (control & (quarter == period)).any():
                    break
            else:
                yt_treat = y[treated_g & (quarter == t)].mean()
                yb_treat = y[treated_g & (quarter == base)].mean()
                yt_ctrl = y[control & (quarter == t)].mean()
                yb_ctrl = y[control & (quarter == base)].mean()
                att_gt = (yt_treat - yb_treat) - (yt_ctrl - yb_ctrl)
                weight = int((treated_g & (quarter == t)).sum())
                records.append((g, t, float(att_gt), weight))
                weighted_sum += att_gt * weight
                total_weight += weight
    att = weighted_sum / total_weight if total_weight > 0 else 0.0
    return att, len(records), records


def _event_study(
    y: np.ndarray, quarter: np.ndarray, treatment_quarter: np.ndarray, window: int
) -> tuple[list[int], list[float]]:
    """Event-time means relative to the not-yet-treated pool (reference e = -1)."""
    treated = treatment_quarter > 0
    event_time = np.where(treated, quarter - treatment_quarter, np.iinfo(np.int64).min)
    control = treatment_quarter <= 0
    ctrl_mean = y[control].mean() if control.any() else y.mean()
    ref = y[treated & (event_time == -1)]
    ref_gap = (ref.mean() - ctrl_mean) if ref.size else 0.0
    times: list[int] = []
    coefs: list[float] = []
    for k in range(-window, window + 1):
        if k == -1:
            continue
        cell = y[treated & (event_time == k)]
        if cell.size:
            times.append(k)
            coefs.append(float((cell.mean() - ctrl_mean) - ref_gap))
    return times, coefs


def estimate_did(
    panel: pl.DataFrame, *, event_window: int = 4, n_boot: int = 200, seed: int = 0
) -> DidResult:
    """Callaway-Sant'Anna-style onset ATT with a firm-cluster bootstrap CI."""
    frame = panel.select("firm_id", "quarter", "outcome_reported", "treatment_quarter")
    y = frame["outcome_reported"].to_numpy().astype(np.float64)
    firm = frame["firm_id"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    cohort = frame["treatment_quarter"].to_numpy().astype(np.int64)

    att, n_cells, _ = group_time_att(y, quarter, cohort)
    event_times, event_coefficients = _event_study(y, quarter, cohort, event_window)
    pre = [abs(c) for t, c in zip(event_times, event_coefficients, strict=True) if t < 0]

    # Firm-cluster bootstrap: resample firms with replacement, recompute the ATT.
    rng = np.random.default_rng(seed)
    unique_firms = np.unique(firm)
    rows_by_firm = {int(f): np.flatnonzero(firm == f) for f in unique_firms}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique_firms, size=unique_firms.size, replace=True)
        idx = np.concatenate([rows_by_firm[int(f)] for f in drawn])
        att_b, _, _ = group_time_att(y[idx], quarter[idx], cohort[idx])
        boot[b] = att_b
    se = float(np.std(boot, ddof=1))
    ci_low, ci_high = (float(v) for v in np.quantile(boot, [0.025, 0.975]))
    return DidResult(
        att=att,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        event_times=event_times,
        event_coefficients=event_coefficients,
        pretrend_max_abs=max(pre) if pre else 0.0,
        n_cells=n_cells,
        n_obs=int(y.size),
    )
