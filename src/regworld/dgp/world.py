"""Entity generation and the true parameters theta* (§7.1, §7.3).

theta* is defined HERE and written only to artifacts/oracle/theta_star.json.
The behavioral truth is identical across DGP variants ("theta is shared across
regimes"); variants differ in structure only (homophily, corr(z, size), decision rule).
"""

from __future__ import annotations

import numpy as np

from regworld.rules import Constants, FirmAttributes, SegmentAttributes, Theta
from regworld.types import RegWorldConfig

THETA_STAR = Theta(
    beta_0=-1.2,
    beta_enforce=2.5,
    beta_cost=1.8,
    beta_peer=1.4,
    beta_assoc=0.6,
    beta_size=0.25,
    beta_customer=0.9,
    phi_phase=0.6,
    beta_stick=2.0,
    beta_capacity=0.9,
    q0=0.05,
    q1=0.05,
    gamma_scale=0.45,
    ell_learn=0.30,
    alpha_trust=0.30,
    rho_influence=0.15,
    mu_privacy=0.80,
    delta_exit=0.25,
)
CONSTANTS = Constants()

# skewed sector distribution (§7.1)
SECTOR_PROBS = np.array([0.30, 0.25, 0.15, 0.12, 0.10, 0.08])
SECTOR_DATA_SHIFT = np.array([0.00, 0.05, -0.05, 0.10, -0.10, 0.00])
SECTOR_COST_MULT = np.array([1.0, 1.2, 0.8, 1.4, 0.7, 1.0])


def generate_firms(cfg: RegWorldConfig, rng: np.random.Generator) -> FirmAttributes:
    n = cfg.population.n_firms
    k = cfg.population.n_sectors
    probs = SECTOR_PROBS[:k] / SECTOR_PROBS[:k].sum()

    size = rng.lognormal(mean=0.0, sigma=1.1, size=n)
    size = size / np.median(size)  # normalized so median = 1
    sector = rng.choice(k, size=n, p=probs)
    data_intensity = np.clip(rng.beta(2, 2, size=n) + SECTOR_DATA_SHIFT[sector], 0.02, 0.98)
    cost_coef = rng.gamma(shape=2.0, scale=0.5, size=n) * SECTOR_COST_MULT[sector]
    quality = rng.normal(0.0, 1.0, size=n)
    base_margin = rng.beta(5, 20, size=n) * 0.5

    # latent capacity z: UNOBSERVED confounder, correlated with log size (§7.7)
    rho = cfg.dgp.corr_z_size
    logs = np.log(size)
    logs_std = (logs - logs.mean()) / max(logs.std(), 1e-9)
    z = rho * logs_std + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.normal(0.0, 1.0, size=n)

    # association membership: probability rising with size; unaffiliated = -1 (§7.1)
    p_member = 1.0 / (1.0 + np.exp(-(0.8 + 0.5 * logs)))
    member = rng.random(n) < p_member
    assoc_by_sector = sector % cfg.population.n_associations
    random_assoc = rng.integers(0, cfg.population.n_associations, size=n)
    mixed = np.where(rng.random(n) < 0.8, assoc_by_sector, random_assoc)
    association = np.where(member, mixed, -1)

    # region assignment: uniform, INDEPENDENT of firm characteristics — this exogeneity
    # is what identifies the staggered-rollout DiD (§7.8)
    region = rng.integers(0, cfg.population.n_regions, size=n)

    terciles = np.quantile(size, [1 / 3, 2 / 3])
    size_tercile = np.digitize(size, terciles)

    return FirmAttributes(
        size=size,
        sector=sector.astype(np.int64),
        data_intensity=data_intensity,
        cost_coef=cost_coef,
        quality=quality,
        base_margin=base_margin,
        z=z,
        association=association.astype(np.int64),
        region=region.astype(np.int64),
        size_tercile=size_tercile.astype(np.int64),
    )


def generate_segments(
    cfg: RegWorldConfig, firms: FirmAttributes, rng: np.random.Generator
) -> tuple[SegmentAttributes, np.ndarray]:
    s = cfg.population.n_consumer_segments
    weight = rng.dirichlet(np.full(s, 5.0))
    privacy = rng.beta(2, 3, size=s)
    budget = weight * float(np.sum(firms.size))  # sum(budget) = sum(size): mean revenue ~ size
    trust0 = rng.beta(5, 3, size=s)
    seg_pref = rng.dirichlet(np.full(cfg.population.n_sectors, 2.0), size=s)  # (S, K)
    return SegmentAttributes(weight=weight, privacy=privacy, budget=budget, trust0=trust0), seg_pref


def theta_star_dict() -> dict[str, float]:
    t = THETA_STAR
    return {name: float(getattr(t, name)) for name in t.group_a_names() + t.group_b_names()} | {
        "beta_capacity": t.beta_capacity
    }
