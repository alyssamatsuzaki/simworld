"""The observation model (§7.9): measurement error, lags, missingness, sampling.

Degrades Regime P's ground truth into what a real corpus would contain. The reporting
lag and misclassification are IN the data — Stage 4 must model them or eat the bias.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from regworld.dgp.dynamics import Trajectory
from regworld.dgp.history import NEVER_TREATED
from regworld.dgp.world import THETA_STAR
from regworld.rules import FirmAttributes, SegmentAttributes
from regworld.types import RegWorldConfig

REVENUE_NOISE_SD = 0.10  # lognormal sd on reported revenue
COST_INDEX_NOISE_SD = 0.40  # registry cost proxy noise (§7.9)
TRUST_REPORT_SD = 0.05


def _flip(y: np.ndarray, q0: float, q1: float, rng: np.random.Generator) -> np.ndarray:
    """Misclassify: P(report 1 | true 0) = q0, P(report 0 | true 1) = q1."""
    u = rng.random(y.shape)
    reported = np.where(y > 0.5, (u >= q1).astype(np.float64), (u < q0).astype(np.float64))
    return reported


def firm_registry(firms: FirmAttributes, rng: np.random.Generator) -> pl.DataFrame:
    """All firms: coarse size, sector, association, noisy cost proxy. NO z, NO true c."""
    deciles = np.quantile(firms.size, np.linspace(0.1, 0.9, 9))
    size_decile = np.digitize(firms.size, deciles)
    cost_index = firms.cost_coef + rng.normal(0.0, COST_INDEX_NOISE_SD, size=firms.n)
    return pl.DataFrame(
        {
            "firm_id": np.arange(firms.n, dtype=np.int64),
            "sector": firms.sector,
            "size_decile": size_decile.astype(np.int64),
            "data_intensity": firms.data_intensity,
            "association": firms.association,
            "cost_index": cost_index,
        }
    )


def firm_panel(
    cfg: RegWorldConfig,
    traj: Trajectory,
    firms: FirmAttributes,
    t_start: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """A `panel_sample_frac` sample of firms, quarters 1..observed_quarters.

    reported_compliant at quarter q reflects the true state at q-1 (one-quarter
    reporting lag) with q0/q1 misclassification. Decision-time covariates are dated
    at the decision quarter itself (what the firm faced when it chose).
    """
    n = firms.n
    n_sample = max(int(round(cfg.dgp.panel_sample_frac * n)), 10)
    sampled = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
    frames = []
    q0, q1 = cfg.dgp.misclassification, cfg.dgp.misclassification
    assert abs(q0 - THETA_STAR.q0) < 1e-9, "config and theta* misclassification must agree"
    for t in range(cfg.observed_quarters):
        covs = traj.covariates[t]
        y_prev = covs["compliant_lag"]  # true y at t-1
        reported = _flip(y_prev, q0, q1, rng)
        revenue_noisy = covs["revenue"] * np.exp(
            rng.normal(0.0, REVENUE_NOISE_SD, size=n)
        )
        frames.append(
            pl.DataFrame(
                {
                    "firm_id": sampled,
                    "quarter": np.full(sampled.size, t + 1, dtype=np.int64),
                    "region": firms.region[sampled],
                    "treatment_quarter": np.where(
                        t_start[sampled] >= NEVER_TREATED, -1, t_start[sampled] + 1
                    ).astype(np.int64),
                    "reported_compliant": reported[sampled],
                    "revenue_noisy": revenue_noisy[sampled],
                    "audited": covs["audited"][sampled].astype(bool),
                    "fined": covs["fined"][sampled].astype(bool),
                    "alive": covs["alive"][sampled].astype(bool),
                    "perceived_risk": covs["perceived_risk"][sampled],
                    "cost_share": covs["cost_share"][sampled],
                    "neighbor_compliant_share": covs["neighbor_compliant_share"][sampled],
                    "assoc_compliant_share": covs["assoc_compliant_share"][sampled],
                    "privacy_rev_share": covs["privacy_rev_share"][sampled],
                    "phase_phi": covs["phase_phi"][sampled],
                    "compliant_lag": covs["compliant_lag"][sampled],
                }
            )
        )
    return pl.concat(frames)


def aggregate_series(
    cfg: RegWorldConfig, traj: Trajectory, rng: np.random.Generator
) -> pl.DataFrame:
    rows = []
    s = cfg.dgp.sigma_obs
    for t in range(cfg.observed_quarters):
        o = traj.outcomes[t]
        rows.append(
            {
                "quarter": t + 1,
                "compliance_rate_obs": o.compliance_rate + rng.normal(0, s),
                "compliance_rate_weighted_obs": o.compliance_rate_weighted + rng.normal(0, s),
                "hhi_obs": o.hhi * (1 + rng.normal(0, s)),
                "mean_trust_obs": o.mean_trust + rng.normal(0, s),
                "exit_rate_obs": o.exit_rate_cum + rng.normal(0, s),
            }
        )
    return pl.DataFrame(rows)


def consumer_survey(
    cfg: RegWorldConfig,
    traj: Trajectory,
    segments: SegmentAttributes,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Quarterly, `survey_sample_frac` of segments, nonresponse rising with privacy
    sensitivity — a selection problem planted on purpose (§7.9)."""
    s = segments.weight.size
    privacy_bucket = np.digitize(segments.privacy, [0.33, 0.66])
    rows = []
    for t in range(cfg.observed_quarters):
        sampled = rng.choice(s, size=max(int(cfg.dgp.survey_sample_frac * s), 2), replace=False)
        p_respond = np.clip(0.95 - 0.5 * segments.privacy[sampled], 0.2, 0.95)
        responded = sampled[rng.random(sampled.size) < p_respond]
        seg_trust = traj.covariates[t]["segment_trust"]
        for j in responded:
            rows.append(
                {
                    "segment_id": int(j),
                    "quarter": t + 1,
                    "trust_reported": float(
                        np.clip(seg_trust[j] + rng.normal(0, TRUST_REPORT_SD), 0, 1)
                    ),
                    "privacy_bucket": int(privacy_bucket[j]),
                }
            )
    return pl.DataFrame(rows)


def market_stats(cfg: RegWorldConfig, traj: Trajectory, firms: FirmAttributes) -> pl.DataFrame:
    rows = []
    for t in range(cfg.observed_quarters):
        rev = traj.covariates[t]["revenue"]
        total = max(float(rev.sum()), 1e-9)
        for k in range(cfg.population.n_sectors):
            share = float(rev[firms.sector == k].sum() / total)
            rows.append({"quarter": t + 1, "sector": k, "revenue_share_rounded": round(share, 3)})
    return pl.DataFrame(rows)


def regime_p_full(
    cfg: RegWorldConfig, traj: Trajectory, firms: FirmAttributes, t_start: np.ndarray
) -> pl.DataFrame:
    """ORACLE ONLY: quarters 1..24, all firms, no noise (§8)."""
    frames = []
    for t in range(cfg.horizon_quarters):
        covs = traj.covariates[t]
        frames.append(
            pl.DataFrame(
                {
                    "firm_id": np.arange(firms.n, dtype=np.int64),
                    "quarter": np.full(firms.n, t + 1, dtype=np.int64),
                    "compliant": covs["compliant"],
                    "alive": covs["alive"].astype(bool),
                    "audited": covs["audited"].astype(bool),
                    "revenue": covs["revenue"],
                    "region": firms.region,
                    "treatment_quarter": np.where(
                        t_start >= NEVER_TREATED, -1, t_start + 1
                    ).astype(np.int64),
                    "size": firms.size,
                    "capacity_z": firms.z,
                    "perceived_risk": covs["perceived_risk"],
                }
            )
        )
    return pl.concat(frames)
