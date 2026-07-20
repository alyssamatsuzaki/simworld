"""Exact observed-panel likelihood for firm compliance decisions (NumPyro)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from regworld.types import RegWorldConfig

MICRO_PARAMETER_NAMES = (
    "beta_0",
    "beta_enforce",
    "beta_cost",
    "beta_peer",
    "beta_assoc",
    "beta_size",
    "beta_customer",
    "phi_phase",
    "beta_stick",
    "q0",
    "q1",
)


@dataclass(frozen=True)
class MicroData:
    """Decision-aligned observed rows consumed by both Bayesian engines."""

    outcome: np.ndarray
    perceived_risk: np.ndarray
    cost_share: np.ndarray
    neighbor_share: np.ndarray
    association_share: np.ndarray
    privacy_share: np.ndarray
    phase: np.ndarray
    compliant_lag: np.ndarray
    log_size: np.ndarray
    sector: np.ndarray
    quarter: np.ndarray
    n_sectors: int

    @property
    def n(self) -> int:
        return int(self.outcome.size)

    def subset(self, mask: np.ndarray) -> MicroData:
        fields = {
            name: getattr(self, name)[mask]
            for name in (
                "outcome",
                "perceived_risk",
                "cost_share",
                "neighbor_share",
                "association_share",
                "privacy_share",
                "phase",
                "compliant_lag",
                "log_size",
                "sector",
                "quarter",
            )
        }
        return MicroData(**fields, n_sectors=self.n_sectors)


def micro_data_from_frame(frame: pl.DataFrame) -> MicroData:
    """Validate and convert the Stage-1 observed analysis panel."""
    required = {
        "outcome_reported",
        "perceived_risk",
        "cost_share",
        "neighbor_compliant_share",
        "assoc_compliant_share",
        "privacy_rev_share",
        "phase_phi",
        "compliant_lag",
        "log_size_proxy",
        "sector",
        "quarter",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"analysis panel missing calibration columns: {sorted(missing)}")
    clean = frame.drop_nulls(sorted(required)).sort(["firm_id", "quarter"])
    if clean.is_empty():
        raise ValueError("analysis panel has no complete calibration rows")
    sector_values = sorted(int(x) for x in clean["sector"].unique().to_list())
    sector_map = {value: index for index, value in enumerate(sector_values)}
    sector = np.asarray([sector_map[int(x)] for x in clean["sector"]], dtype=np.int32)

    def values(name: str) -> np.ndarray:
        return clean[name].to_numpy().astype(np.float32, copy=False)

    outcome = values("outcome_reported")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("reported compliance must be binary")
    return MicroData(
        outcome=outcome,
        perceived_risk=values("perceived_risk"),
        cost_share=values("cost_share"),
        neighbor_share=values("neighbor_compliant_share"),
        association_share=values("assoc_compliant_share"),
        privacy_share=values("privacy_rev_share"),
        phase=values("phase_phi"),
        compliant_lag=values("compliant_lag"),
        log_size=values("log_size_proxy"),
        sector=sector,
        quarter=clean["quarter"].to_numpy().astype(np.int32, copy=False),
        n_sectors=len(sector_values),
    )


def load_micro_data(cfg: RegWorldConfig) -> MicroData:
    """Load only the analyst-created panel; no hidden simulation state is accepted."""
    path = Path(cfg.paths.data) / "panel_analysis.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"analysis panel not found: {path}; run `make data` first")
    return micro_data_from_frame(pl.read_parquet(path))


def reported_probability(true_probability: np.ndarray, q0: float, q1: float) -> np.ndarray:
    """Marginal report probability after false-positive/false-negative error."""
    probability = np.asarray(true_probability, dtype=np.float64)
    if not (0.0 <= q0 <= 1.0 and 0.0 <= q1 <= 1.0):
        raise ValueError("misclassification rates must lie in [0,1]")
    return q0 + (1.0 - q0 - q1) * probability


def micro_model(data: MicroData, *, observe: bool = True) -> None:
    """Hierarchical firm-decision likelihood with integrated report error.

    The latent Bernoulli compliance state is marginalized analytically. This is
    exact and avoids sampling one discrete variable per firm-quarter.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 2.0))
    beta_enforce = numpyro.sample("beta_enforce", dist.HalfNormal(2.0))
    beta_cost = numpyro.sample("beta_cost", dist.HalfNormal(2.0))
    beta_peer = numpyro.sample("beta_peer", dist.Normal(1.0, 1.0))
    beta_assoc = numpyro.sample("beta_assoc", dist.Normal(0.5, 1.0))
    beta_size = numpyro.sample("beta_size", dist.Normal(0.0, 1.0))
    beta_customer = numpyro.sample("beta_customer", dist.HalfNormal(1.0))
    phi_phase = numpyro.sample("phi_phase", dist.Normal(0.5, 0.5))
    beta_stick = numpyro.sample("beta_stick", dist.HalfNormal(1.0))
    q0 = numpyro.sample("q0", dist.Beta(2.0, 20.0))
    q1 = numpyro.sample("q1", dist.Beta(2.0, 20.0))

    beta0_scale = numpyro.sample("beta0_sector_scale", dist.HalfNormal(0.5))
    cost_scale = numpyro.sample("beta_cost_sector_scale", dist.HalfNormal(0.35))
    beta0_offset = numpyro.sample(
        "beta0_sector_offset", dist.Normal(0.0, 1.0).expand([data.n_sectors])
    )
    cost_offset = numpyro.sample(
        "beta_cost_sector_offset", dist.Normal(0.0, 1.0).expand([data.n_sectors])
    )
    sector = jnp.asarray(data.sector)
    beta0_local = beta_0 + beta0_scale * beta0_offset[sector]
    # Multiplicative pooling keeps every sector's cost sensitivity positive.
    beta_cost_local = beta_cost * jnp.exp(cost_scale * cost_offset[sector])
    eta = (
        beta0_local
        + beta_enforce * jnp.asarray(data.perceived_risk)
        - beta_cost_local * jnp.asarray(data.cost_share)
        + beta_peer * jnp.asarray(data.neighbor_share)
        + beta_assoc * jnp.asarray(data.association_share)
        + beta_size * jnp.asarray(data.log_size)
        + beta_customer * jnp.asarray(data.privacy_share)
        + phi_phase * jnp.asarray(data.phase)
        - beta_stick * (1.0 - jnp.asarray(data.compliant_lag))
    )
    true_probability = jnp.asarray(1.0 / (1.0 + jnp.exp(-eta)))
    report_probability = q0 + (1.0 - q0 - q1) * true_probability
    numpyro.deterministic("mean_true_compliance", true_probability.mean())
    numpyro.sample(
        "reported_compliance",
        dist.Bernoulli(probs=jnp.clip(report_probability, 1e-6, 1.0 - 1e-6)),
        obs=jnp.asarray(data.outcome) if observe else None,
    )


def fit_micro_numpyro(
    data: MicroData,
    *,
    seed: int,
    warmup: int,
    draws: int,
    chains: int,
    target_accept: float = 0.9,
) -> tuple[Any, Any]:
    """Run NUTS and return ``(InferenceData, MCMC)``.

    Imports of JAX and NumPyro are deliberately function-local so the pipeline
    driver can launch this function in an isolated CPU process.
    """
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS, Predictive

    if data.n < 20:
        raise ValueError("micro calibration needs at least 20 observed decisions")
    kernel = NUTS(micro_model, target_accept_prob=target_accept, dense_mass=False)
    mcmc = MCMC(
        kernel,
        num_warmup=int(warmup),
        num_samples=int(draws),
        num_chains=int(chains),
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        data=data,
        observe=True,
        extra_fields=("potential_energy", "energy", "num_steps", "accept_prob"),
    )
    flat_samples = mcmc.get_samples(group_by_chain=False)
    predictive = Predictive(micro_model, posterior_samples=flat_samples)
    ppc = predictive(jax.random.PRNGKey(seed + 1), data=data, observe=False)
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive={"reported_compliance": ppc["reported_compliance"]},
    )
    if "observed_data" not in idata.groups():
        idata.extend(az.from_dict(observed_data={"reported_compliance": data.outcome}))
    return idata, mcmc


def _tiny_model(design: np.ndarray, successes: np.ndarray | None, trials: int) -> None:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 2.0))
    beta_enforce = numpyro.sample("beta_enforce", dist.HalfNormal(2.0))
    beta_peer = numpyro.sample("beta_peer", dist.Normal(1.0, 1.0))
    eta = beta_0 + beta_enforce * jnp.asarray(design[:, 0]) + beta_peer * jnp.asarray(design[:, 1])
    numpyro.sample(
        "successes",
        dist.Binomial(total_count=trials, logits=eta),
        obs=None if successes is None else jnp.asarray(successes),
    )


def fit_tiny_numpyro(
    design: np.ndarray,
    successes: np.ndarray,
    *,
    trials: int,
    seed: int = 0,
    warmup: int = 200,
    draws: int = 200,
    chains: int = 2,
) -> Any:
    """Small three-parameter recovery harness used by the fast scientific gate."""
    import arviz as az
    import jax
    from numpyro.infer import MCMC, NUTS

    design = np.asarray(design, dtype=np.float32)
    successes = np.asarray(successes, dtype=np.int32)
    if design.ndim != 2 or design.shape[1] != 2 or design.shape[0] != successes.size:
        raise ValueError("design must be (N,2) and align with successes")
    mcmc = MCMC(
        NUTS(_tiny_model, target_accept_prob=0.9),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        design=design,
        successes=successes,
        trials=int(trials),
        extra_fields=("potential_energy", "energy", "num_steps", "accept_prob"),
    )
    return az.from_numpyro(mcmc)
