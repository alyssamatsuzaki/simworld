"""Independent PyMC implementation of the observed-panel micro likelihood."""

from __future__ import annotations

from typing import Any

import numpy as np

from regworld.calibration.micro_numpyro import MICRO_PARAMETER_NAMES, MicroData


def fit_micro_pymc(
    data: MicroData,
    *,
    seed: int,
    warmup: int,
    draws: int,
    chains: int,
    target_accept: float = 0.9,
) -> Any:
    """Sample the same marginalized report-error model with PyMC's NUTS."""
    import pymc as pm

    if data.n < 20:
        raise ValueError("micro calibration needs at least 20 observed decisions")
    with pm.Model():
        perceived_risk = pm.Data("perceived_risk", data.perceived_risk)
        cost_share = pm.Data("cost_share", data.cost_share)
        neighbor_share = pm.Data("neighbor_share", data.neighbor_share)
        association_share = pm.Data("association_share", data.association_share)
        privacy_share = pm.Data("privacy_share", data.privacy_share)
        phase = pm.Data("phase", data.phase)
        compliant_lag = pm.Data("compliant_lag", data.compliant_lag)
        log_size = pm.Data("log_size", data.log_size)
        sector = pm.Data("sector", data.sector)

        beta_0 = pm.Normal("beta_0", 0.0, 2.0)
        beta_enforce = pm.HalfNormal("beta_enforce", 2.0)
        beta_cost = pm.HalfNormal("beta_cost", 2.0)
        beta_peer = pm.Normal("beta_peer", 1.0, 1.0)
        beta_assoc = pm.Normal("beta_assoc", 0.5, 1.0)
        beta_size = pm.Normal("beta_size", 0.0, 1.0)
        beta_customer = pm.HalfNormal("beta_customer", 1.0)
        phi_phase = pm.Normal("phi_phase", 0.5, 0.5)
        beta_stick = pm.HalfNormal("beta_stick", 1.0)
        q0 = pm.Beta("q0", 2.0, 20.0)
        q1 = pm.Beta("q1", 2.0, 20.0)

        beta0_scale = pm.HalfNormal("beta0_sector_scale", 0.5)
        cost_scale = pm.HalfNormal("beta_cost_sector_scale", 0.35)
        beta0_offset = pm.Normal("beta0_sector_offset", 0.0, 1.0, shape=data.n_sectors)
        cost_offset = pm.Normal("beta_cost_sector_offset", 0.0, 1.0, shape=data.n_sectors)
        beta0_local = beta_0 + beta0_scale * beta0_offset[sector]
        beta_cost_local = beta_cost * pm.math.exp(cost_scale * cost_offset[sector])
        eta = (
            beta0_local
            + beta_enforce * perceived_risk
            - beta_cost_local * cost_share
            + beta_peer * neighbor_share
            + beta_assoc * association_share
            + beta_size * log_size
            + beta_customer * privacy_share
            + phi_phase * phase
            - beta_stick * (1.0 - compliant_lag)
        )
        true_probability = pm.math.sigmoid(eta)
        report_probability = q0 + (1.0 - q0 - q1) * true_probability
        pm.Deterministic("mean_true_compliance", pm.math.mean(true_probability))
        pm.Bernoulli(
            "reported_compliance",
            p=pm.math.clip(report_probability, 1e-6, 1.0 - 1e-6),
            observed=data.outcome.astype(np.int8),
        )
        idata = pm.sample(
            tune=int(warmup),
            draws=int(draws),
            chains=int(chains),
            cores=1,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=False,
            return_inferencedata=True,
        )
        pm.sample_posterior_predictive(
            idata,
            var_names=["reported_compliance"],
            random_seed=seed + 1,
            progressbar=False,
            extend_inferencedata=True,
        )
    return idata


def compare_marginals(primary: Any, crosscheck: Any) -> dict[str, Any]:
    """Compare scalar marginals using overlap and standardized mean shifts."""
    rows: dict[str, dict[str, float | bool]] = {}
    for name in MICRO_PARAMETER_NAMES:
        first = np.asarray(primary.posterior[name]).reshape(-1)
        second = np.asarray(crosscheck.posterior[name]).reshape(-1)
        first_interval = np.quantile(first, [0.05, 0.95])
        second_interval = np.quantile(second, [0.05, 0.95])
        pooled_sd = float(np.sqrt(0.5 * (np.var(first, ddof=1) + np.var(second, ddof=1))))
        standardized = abs(float(first.mean() - second.mean())) / max(pooled_sd, 1e-12)
        overlap = bool(
            max(float(first_interval[0]), float(second_interval[0]))
            <= min(float(first_interval[1]), float(second_interval[1]))
        )
        rows[name] = {
            "primary_mean": float(first.mean()),
            "crosscheck_mean": float(second.mean()),
            "pooled_sd": pooled_sd,
            "standardized_mean_difference": standardized,
            "credible_intervals_overlap": overlap,
        }
    return {
        "parameters": rows,
        "all_90pct_intervals_overlap": all(
            bool(row["credible_intervals_overlap"]) for row in rows.values()
        ),
        "all_mean_differences_below_0_1_sd": all(
            float(row["standardized_mean_difference"]) < 0.1 for row in rows.values()
        ),
    }
