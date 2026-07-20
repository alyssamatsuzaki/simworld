"""ArviZ diagnostics and predictive checks for Stage 4 calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from regworld.calibration.micro_numpyro import MICRO_PARAMETER_NAMES, MicroData, micro_model


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _predictive_checks(
    mcmc: Any,
    train: MicroData,
    heldout: MicroData,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    import jax
    from numpyro.infer import Predictive

    prior_draws = Predictive(micro_model, num_samples=300)(
        jax.random.PRNGKey(seed + 10), data=train, observe=False
    )["reported_compliance"]
    prior_rates = np.asarray(prior_draws).mean(axis=1)
    observed_rate = float(train.outcome.mean())
    prior_interval = np.quantile(prior_rates, [0.01, 0.99])
    prior_ok = bool(prior_interval[0] <= observed_rate <= prior_interval[1])

    plt.hist(prior_rates, bins=30, alpha=0.75, color="#5276A7")
    plt.axvline(observed_rate, color="#A33A3A", linewidth=2, label="observed")
    plt.xlabel("mean reported compliance")
    plt.ylabel("prior predictive draws")
    plt.legend()
    _save_current(output_dir / "prior_predictive.png")

    samples = mcmc.get_samples(group_by_chain=False)
    heldout_draws = Predictive(micro_model, posterior_samples=samples)(
        jax.random.PRNGKey(seed + 11), data=heldout, observe=False
    )["reported_compliance"]
    heldout_array = np.asarray(heldout_draws, dtype=np.float64)
    quarters = np.unique(heldout.quarter)
    predicted_mean = []
    predicted_low = []
    predicted_high = []
    actual = []
    for quarter in quarters:
        mask = heldout.quarter == quarter
        rates = heldout_array[:, mask].mean(axis=1)
        predicted_mean.append(float(rates.mean()))
        predicted_low.append(float(np.quantile(rates, 0.05)))
        predicted_high.append(float(np.quantile(rates, 0.95)))
        actual.append(float(heldout.outcome[mask].mean()))
    plt.plot(quarters, actual, marker="o", label="held-out observed", color="#222222")
    plt.plot(quarters, predicted_mean, marker="o", label="posterior predictive", color="#5276A7")
    plt.fill_between(quarters, predicted_low, predicted_high, alpha=0.25, color="#5276A7")
    plt.ylim(0.0, 1.0)
    plt.xlabel("quarter")
    plt.ylabel("reported compliance")
    plt.legend()
    _save_current(output_dir / "posterior_predictive_heldout.png")
    row_probability = heldout_array.mean(axis=0)
    brier = float(np.mean((row_probability - heldout.outcome) ** 2))
    return {
        "prior_predictive": {
            "observed_rate": observed_rate,
            "central_98pct_interval": prior_interval.tolist(),
            "passed": prior_ok,
        },
        "heldout_posterior_predictive": {
            "quarters": quarters.astype(int).tolist(),
            "observed_rates": actual,
            "predicted_rates": predicted_mean,
            "brier_score": brier,
        },
    }


def run_micro_diagnostics(
    idata: Any,
    mcmc: Any,
    train: MicroData,
    heldout: MicroData,
    *,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Write numeric diagnostics plus energy, pair, and predictive plots."""
    import arviz as az

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = az.summary(
        idata,
        var_names=list(MICRO_PARAMETER_NAMES),
        hdi_prob=0.90,
        kind="all",
    )
    summary_path = output_dir / "micro_posterior_summary.csv"
    summary.to_csv(summary_path)
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    numeric = {
        name: {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_5pct": float(row["hdi_5%"]),
            "hdi_95pct": float(row["hdi_95%"]),
            "r_hat": float(row["r_hat"]),
            "ess_bulk": float(row["ess_bulk"]),
            "ess_tail": float(row["ess_tail"]),
        }
        for name, row in summary.iterrows()
    }
    payload: dict[str, Any] = {
        "parameters": numeric,
        "divergences": divergences,
        "max_r_hat": max(row["r_hat"] for row in numeric.values()),
        "min_ess_bulk": min(row["ess_bulk"] for row in numeric.values()),
        "min_ess_tail": min(row["ess_tail"] for row in numeric.values()),
    }
    payload.update(_predictive_checks(mcmc, train, heldout, seed=seed, output_dir=output_dir))
    if not payload["prior_predictive"]["passed"]:
        raise RuntimeError("prior predictive distribution cannot reproduce the observed mean")

    az.plot_energy(idata)
    energy_path = output_dir / "energy.png"
    _save_current(energy_path)
    az.plot_pair(
        idata,
        var_names=["beta_peer", "beta_assoc"],
        kind="kde",
        marginals=True,
        divergences=True,
    )
    pair_path = output_dir / "peer_assoc_pair.png"
    _save_current(pair_path)
    diagnostics_path = output_dir / "micro_diagnostics.json"
    diagnostics_path.write_text(json.dumps(payload, indent=2))
    return payload, [
        summary_path,
        diagnostics_path,
        energy_path,
        pair_path,
        output_dir / "prior_predictive.png",
        output_dir / "posterior_predictive_heldout.png",
    ]


def combine_posteriors(micro: Any, macro: Any, *, seed: int) -> Any:
    """Attach independent Group-B draws to the micro posterior draw grid."""
    combined = micro.copy()
    n_chains = int(combined.posterior.sizes["chain"])
    n_draws = int(combined.posterior.sizes["draw"])
    n_needed = n_chains * n_draws
    rng = np.random.default_rng(seed)
    for name in macro.posterior.data_vars:
        values = np.asarray(macro.posterior[name]).reshape(-1)
        selected = rng.choice(values, size=n_needed, replace=values.size < n_needed)
        combined.posterior[name] = (("chain", "draw"), selected.reshape(n_chains, n_draws))
    return combined
