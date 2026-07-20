"""PURE decision functions (§7.4), shared by dgp/ AND abm/.

The equations are written once, here, and never drift: the DGP binds theta = theta*,
the estimated ABM binds theta = a posterior draw. No in-place mutation — every step
function takes (state, theta, policy, rng) and returns new arrays.

Everything is NumPy over the whole firm population; the per-agent Mesa loop delegates
to these same functions (§16 guardrail 14), and abm/tensorized.py re-implements them
in torch under an agreement test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.special import expit, logsumexp


@dataclass(frozen=True)
class Theta:
    """Behavioral parameters (§7.3). Group A: firm logit; Group B: consumer/market.

    Defaults are deliberately ordinary prior-center starting values for the
    estimated model. The planted answer-key values are bound explicitly inside
    ``regworld.dgp.world`` and must never be recoverable by calling ``Theta()``.
    """

    # Group A — firm-decision logit (micro-likelihood, Stage 4a)
    beta_0: float = 0.0
    beta_enforce: float = 1.0
    beta_cost: float = 1.0
    beta_peer: float = 1.0
    beta_assoc: float = 0.5
    beta_size: float = 0.0
    beta_customer: float = 0.5
    phi_phase: float = 0.5
    beta_stick: float = 1.0
    beta_capacity: float = 0.0  # fitted models never observe latent capacity z_i
    q0: float = 2.0 / 22.0  # Beta(2,20) prior mean; truth is bound only in dgp/
    q1: float = 2.0 / 22.0
    # Group B — consumer, market, enforcement dynamics (macro SMC-ABC, Stage 4b)
    gamma_scale: float = 0.50
    ell_learn: float = 1.0 / 3.0
    alpha_trust: float = 2.0 / 7.0
    rho_influence: float = 0.20
    mu_privacy: float = 0.80
    delta_exit: float = 0.40

    def group_a_names(self) -> list[str]:
        return [
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
        ]

    def group_b_names(self) -> list[str]:
        return [
            "gamma_scale",
            "ell_learn",
            "alpha_trust",
            "rho_influence",
            "mu_privacy",
            "delta_exit",
        ]


@dataclass(frozen=True)
class Constants:
    """Fixed / known quantities (§7.3) — never calibrated."""

    audit_budget: float = 0.25  # B: fraction of firms auditable per quarter at e=1
    risk_scale: float = 4.0  # Phi in the perceived-risk equation (utility units)
    fine_rate: float = 0.08  # actual fine as a fraction of revenue
    fine_cap: float = 0.15  # f_cap
    target_gamma: float = 0.5  # gamma: size exponent in the targeting weight
    penalty_psi: float = 0.3  # psi: size exponent in perceived penalty
    publicity_omega: float = 0.5  # omega: publicity multiplier on perceived risk
    publicity_decay: float = 0.7  # EWMA decay for sector publicity
    cost_scale: float = 0.15  # kappa scale: mean compliance cost ~ a tenth of revenue
    exit_floor: float = 0.4  # xi: revenue floor factor (x baseline firm revenue)
    quality_weight: float = 1.0  # lambda_quality in consumer spend utility
    trust_noise: float = 0.02  # sd of the trust update shock
    spend_utility_noise: float = 0.3  # sd of epsilon_ji in the spend utility
    audit_unit_cost: float = 0.01  # enforcement cost per audit (in mean-revenue units)
    own_audit_boost: float = 0.8  # being audited last quarter raises perceived risk (§7.7 DAG)
    interaction_beta: float = 0.8  # extra q*kappa term under dgp=misspecified only


@dataclass(frozen=True)
class PolicyLevers:
    """The Gymnasium action (§7.5)."""

    enforcement: float = 0.0  # e in [0, 1]
    targeting: float = 0.0  # tau in [-1, 1]
    phase_speed: float = 0.0  # in [0, 1] -> phase-in length L = 12 - 10 * speed
    subsidy: float = 0.0  # in [0, 1], bottom size tercile

    def phase_length(self) -> float:
        return 12.0 - self.phase_speed * 10.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.enforcement, self.targeting, self.phase_speed, self.subsidy], dtype=np.float64
        )


@dataclass(frozen=True)
class FirmAttributes:
    """Static firm attributes (§7.1). `z` is the planted unobserved confounder."""

    size: np.ndarray  # s_i, median 1
    sector: np.ndarray  # k_i in 0..K-1
    data_intensity: np.ndarray  # d_i in [0, 1]
    cost_coef: np.ndarray  # c_i
    quality: np.ndarray  # Q_i
    base_margin: np.ndarray  # m0_i
    z: np.ndarray  # latent capacity — UNOBSERVED downstream
    association: np.ndarray  # a_i in 0..A-1, or -1 if unaffiliated
    region: np.ndarray  # r_i (Regime P rollout unit)
    size_tercile: np.ndarray  # 0 = small, 1 = mid, 2 = large

    @property
    def n(self) -> int:
        return int(self.size.shape[0])


@dataclass(frozen=True)
class SegmentAttributes:
    weight: np.ndarray  # w_j, sums to 1
    privacy: np.ndarray  # p_j in [0, 1]
    budget: np.ndarray  # b_j; sum(budget) = sum(size) so mean firm revenue ~ size
    trust0: np.ndarray  # T_j(0)


@dataclass(frozen=True)
class Graphs:
    """Adjacency structure the dynamics run on (true OR observed — same type)."""

    supply_und: sparse.csr_matrix  # symmetrized supply adjacency (peer term, in+out)
    influence: sparse.csr_matrix  # segment-segment row-normalized influence
    market_mask: np.ndarray  # (S, F) bool: segment j can buy from firm i


@dataclass
class WorldState:
    """Mutable-by-replacement quarterly state."""

    y: np.ndarray  # (F,) 0/1 compliant
    alive: np.ndarray  # (F,) bool
    revenue: np.ndarray  # (F,)
    tenure: np.ndarray  # (F,) quarters compliant so far
    fines: np.ndarray  # (F,) fines paid this quarter
    audited: np.ndarray  # (F,) bool, this quarter
    spend: np.ndarray  # (S, F) consumer spend matrix
    trust: np.ndarray  # (S,)
    publicity: np.ndarray  # (K,) sector-level enforcement salience (EWMA)
    rev_hist: np.ndarray  # (F, 3) rolling revenue window
    below_floor: np.ndarray  # (F,) consecutive quarters under the exit floor
    quarter: int = 0


@dataclass(frozen=True)
class QuarterOutcome:
    """The §7.6 outcome vector, one quarter."""

    compliance_rate: float
    compliance_rate_weighted: float
    compliance_by_tercile: tuple[float, float, float]
    hhi: float
    mean_trust: float
    consumer_surplus: float
    exit_rate_cum: float
    enforcement_cost: float
    n_audits: int


OUTCOME_VARIABLES = [
    "compliance_rate",
    "compliance_rate_weighted",
    "hhi",
    "mean_trust",
    "consumer_surplus",
    "exit_rate",
    "enforcement_cost",
    "reward",
    "backfire",
]


def phase_progress(t: int, levers: PolicyLevers, t_start: np.ndarray | int = 0) -> np.ndarray:
    """phi(t) = min(1, quarters-since-start / L); 0 before enforcement starts."""
    since = np.asarray(t - np.asarray(t_start) + 1, dtype=np.float64)
    phi = np.clip(since / levers.phase_length(), 0.0, 1.0)
    return np.where(since > 0, phi, 0.0)


def audit_probabilities(
    state: WorldState,
    firms: FirmAttributes,
    const: Constants,
    levers: PolicyLevers,
    active: np.ndarray,
) -> np.ndarray:
    """w_it and alpha_it (§7.4). `active` masks firms whose region has enforcement on."""
    tau = abs(levers.targeting)
    # tau > 0 targets large + previously non-compliant; tau < 0 flips to target small firms
    size_term = (
        firms.size**const.target_gamma
        if levers.targeting >= 0
        else ((1.0 / np.maximum(firms.size, 1e-9)) ** const.target_gamma)
    )
    w = (1.0 - tau) + tau * size_term * (1.0 - state.y)
    w = w * state.alive * active
    total = float(np.sum(w))
    if total <= 0.0:
        return np.zeros_like(w)
    n_audits = levers.enforcement * const.audit_budget * firms.n
    return np.clip(n_audits * w / total, 0.0, 1.0)


def perceived_risk(
    alpha: np.ndarray,
    firms: FirmAttributes,
    state: WorldState,
    const: Constants,
    phi: np.ndarray,
) -> np.ndarray:
    """q_it: expected-penalty term entering the utility (§7.4).

    The (1 + own_audit_boost * audited_{t-1}) factor realizes the §7.7 DAG arrow
    audited -> perceived_risk -> compliant_next; it appears in the panel's
    perceived_risk column, so the fitted micro model stays well specified.
    """
    pub = state.publicity[firms.sector]
    return (
        alpha
        * const.risk_scale
        * firms.size**const.penalty_psi
        * phi
        * (1.0 + const.publicity_omega * pub)
        * (1.0 + const.own_audit_boost * state.audited.astype(np.float64))
    )


def compliance_cost_share(
    firms: FirmAttributes,
    state: WorldState,
    theta: Theta,
    const: Constants,
    phi: np.ndarray,
    levers: PolicyLevers,
) -> np.ndarray:
    """kappa_it — the backfire mechanism: economies of scale + learning-by-doing (§7.4)."""
    s_med = 1.0  # sizes are median-normalized at generation
    kappa = (
        const.cost_scale
        * firms.cost_coef
        * firms.data_intensity
        * (firms.size / s_med) ** (-theta.gamma_scale)
        * phi
        * (1.0 - theta.ell_learn * np.minimum(1.0, state.tenure / 12.0))
    )
    kappa = kappa * (1.0 - levers.subsidy * (firms.size_tercile == 0))
    return kappa


def neighbour_share(y_prev: np.ndarray, alive: np.ndarray, graphs: Graphs) -> np.ndarray:
    """n_{i,t-1}: lagged supply-neighbour compliance share, in+out edges (§7.4)."""
    a = graphs.supply_und
    num = a @ (y_prev * alive)
    den = a @ alive.astype(np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def association_share(
    y_prev: np.ndarray, alive: np.ndarray, firms: FirmAttributes, n_assoc: int
) -> np.ndarray:
    """m_{k,t-1}: mean lagged compliance among the firm's association members.

    Unaffiliated firms (association == -1) see the economy-wide mean.
    """
    alive_f = alive.astype(np.float64)
    overall = float(np.sum(y_prev * alive_f) / max(np.sum(alive_f), 1.0))
    out = np.full(firms.n, overall)
    for a in range(n_assoc):
        mask = (firms.association == a) & alive
        if mask.sum() > 0:
            out[firms.association == a] = float(np.mean(y_prev[mask]))
    return out


def privacy_revenue_share(
    spend: np.ndarray, revenue: np.ndarray, segments: SegmentAttributes
) -> np.ndarray:
    """x_{i,t}: privacy-weighted revenue share (§7.4)."""
    num = spend.T @ segments.privacy  # (F,)
    return np.divide(num, revenue, out=np.zeros_like(num), where=revenue > 0)


def firm_utility(
    *,
    theta: Theta,
    const: Constants,
    q: np.ndarray,
    kappa: np.ndarray,
    n_peer: np.ndarray,
    m_assoc: np.ndarray,
    log_size: np.ndarray,
    x_privacy: np.ndarray,
    phi: np.ndarray,
    y_prev: np.ndarray,
    z: np.ndarray | None = None,
    sticky: bool = True,
    interacted: bool = False,
) -> np.ndarray:
    """u_it — THE equation Stage 4a must recover (§7.4).

    `z` is passed only by the DGP (beta_capacity * z is absent from every fitted model).
    `interacted` adds the q*kappa term only under dgp=misspecified.
    """
    u = (
        theta.beta_0
        + theta.beta_enforce * q
        - theta.beta_cost * kappa
        + theta.beta_peer * n_peer
        + theta.beta_assoc * m_assoc
        + theta.beta_size * log_size
        + theta.beta_customer * x_privacy
        + theta.phi_phase * phi
    )
    if z is not None:
        u = u + theta.beta_capacity * z
    if sticky:
        u = u - theta.beta_stick * (1.0 - y_prev)
    if interacted:
        u = u + const.interaction_beta * q * kappa
    return u


def draw_audits(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample ~ floor(sum(alpha)) audits without replacement with probs proportional to alpha."""
    n_audits = int(np.floor(np.sum(alpha)))
    audited = np.zeros(alpha.shape[0], dtype=bool)
    eligible = np.flatnonzero(alpha > 0)
    if n_audits <= 0 or eligible.size == 0:
        return audited
    n_audits = min(n_audits, eligible.size)
    p = alpha[eligible] / np.sum(alpha[eligible])
    chosen = rng.choice(eligible, size=n_audits, replace=False, p=p)
    audited[chosen] = True
    return audited


def step_consumers(
    state: WorldState,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs: Graphs,
    theta: Theta,
    const: Constants,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Trust update + spend reallocation (§7.4).

    The sampled utility matrix is returned so a same-quarter exit can remove firms
    and redistribute demand without drawing a second, inconsistent consumer shock.
    """
    spend_share = state.spend / np.maximum(state.spend.sum(axis=1, keepdims=True), 1e-9)
    exposure = spend_share @ (state.y * state.alive)  # (S,)
    social = graphs.influence @ state.trust - state.trust
    eps = rng.normal(0.0, const.trust_noise, size=state.trust.shape)
    trust = np.clip(
        state.trust
        + theta.alpha_trust * (exposure - state.trust)
        + theta.rho_influence * social
        + eps,
        0.0,
        1.0,
    )
    # spend utility v_ji over alive, linked firms
    v = (
        const.quality_weight * firms.quality[None, :]
        + theta.mu_privacy * segments.privacy[:, None] * (state.y * state.alive)[None, :]
        + rng.normal(0.0, const.spend_utility_noise, size=(segments.weight.size, firms.n))
    )
    spend, revenue, cs = allocate_spend(v, state.alive, graphs.market_mask, segments)
    return trust, spend, revenue, cs, v


def allocate_spend(
    utility: np.ndarray,
    alive: np.ndarray,
    market_mask: np.ndarray,
    segments: SegmentAttributes,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Allocate each segment's budget over its linked, surviving firms.

    Keeping this separate from the stochastic trust/utility update lets the exit
    step re-run the softmax with the *same* utility draw. Thus newly exited firms
    have zero spend and revenue in the quarter's reported market outcomes.
    """
    mask = market_mask & alive[None, :]
    v_masked = np.where(mask, utility, -np.inf)
    # consumer surplus: logit inclusive value, weight-weighted (§7.6)
    lse = logsumexp(v_masked, axis=1)
    lse = np.where(np.isfinite(lse), lse, 0.0)
    cs = float(np.sum(segments.weight * lse))
    vmax = np.max(v_masked, axis=1, keepdims=True)
    vmax = np.where(np.isfinite(vmax), vmax, 0.0)
    expv = np.where(mask, np.exp(utility - vmax), 0.0)
    denom = np.maximum(expv.sum(axis=1, keepdims=True), 1e-12)
    shares = expv / denom
    spend = segments.budget[:, None] * shares
    revenue = spend.sum(axis=0)
    return spend, revenue, cs


def step_market_and_exit(
    state: WorldState,
    firms: FirmAttributes,
    theta: Theta,
    const: Constants,
    kappa: np.ndarray,
    revenue_new: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Margin + exit hazard (§7.4). Returns (alive', rev_hist', below_floor')."""
    rev_safe = np.maximum(revenue_new, 1e-9)
    margin = firms.base_margin - state.y * kappa - state.fines / rev_safe
    rev_hist = np.roll(state.rev_hist, -1, axis=1)
    rev_hist[:, -1] = revenue_new
    filled = min(state.quarter + 1, 3)
    rolling = rev_hist[:, -filled:].mean(axis=1)
    floor = const.exit_floor * firms.size  # baseline revenue ~ size (median-normalized)
    below = np.where(rolling < floor, state.below_floor + 1, 0)
    hazard = np.minimum(
        1.0, theta.delta_exit * np.abs(margin) * (1.0 / np.maximum(firms.size, 1e-9))
    )
    at_risk = state.alive & (below >= 2)
    exits = at_risk & (rng.random(firms.n) < hazard)
    alive = state.alive & ~exits
    return alive, rev_hist, below


def hhi(revenue: np.ndarray, alive: np.ndarray) -> float:
    r = revenue * alive
    total = r.sum()
    if total <= 0:
        return 0.0
    shares = r / total
    return float(10_000.0 * np.sum(shares**2))


def step_quarter(
    state: WorldState,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs: Graphs,
    theta: Theta,
    const: Constants,
    levers: PolicyLevers,
    rng: np.random.Generator,
    *,
    t_start: np.ndarray | int = 0,
    use_z: bool = False,
    sticky: bool = True,
    interacted: bool = False,
    attention: float = 1.0,
) -> tuple[WorldState, QuarterOutcome, dict[str, np.ndarray]]:
    """One quarter of the world (§7.4), shared verbatim by the DGP and the ABM.

    Returns (next state, outcome vector, decision-time covariates for the panel).
    `use_z=True` only in the DGP: the beta_capacity * z term exists in the world,
    never in the fitted model.
    """
    t = state.quarter
    active = (np.asarray(t_start) <= t).astype(np.float64)
    phi = phase_progress(t, levers, t_start)

    alpha = audit_probabilities(state, firms, const, levers, active)
    q = perceived_risk(alpha, firms, state, const, phi)
    kappa = compliance_cost_share(firms, state, theta, const, phi, levers)
    n_peer = neighbour_share(state.y, state.alive.astype(np.float64), graphs)
    m_assoc = association_share(state.y, state.alive, firms, int(firms.association.max()) + 1)
    x_priv = privacy_revenue_share(state.spend, state.revenue, segments)
    log_size = np.log(np.maximum(firms.size, 1e-9))

    u = firm_utility(
        theta=theta,
        const=const,
        q=q,
        kappa=kappa,
        n_peer=n_peer,
        m_assoc=m_assoc,
        log_size=log_size,
        x_privacy=x_priv,
        phi=phi,
        y_prev=state.y,
        z=firms.z if use_z else None,
        sticky=sticky,
        interacted=interacted,
    )
    y_prob = expit(u)
    draws = rng.random(firms.n)
    y_new = (draws < y_prob).astype(np.float64)
    if attention < 1.0:  # bounded rationality: some firms do not reconsider this quarter
        reconsider = rng.random(firms.n) < attention
        y_new = np.where(reconsider, y_new, state.y)
    y_new = y_new * state.alive

    audited = draw_audits(alpha, rng)
    caught = audited & (y_new < 0.5) & state.alive
    fines = np.where(
        caught, np.minimum(const.fine_rate, const.fine_cap) * np.maximum(state.revenue, 0.0), 0.0
    )
    # sector publicity: EWMA of sector fines normalized by sector revenue
    n_sectors = int(firms.sector.max()) + 1
    sector_fines = np.bincount(firms.sector, weights=fines, minlength=n_sectors)
    sector_rev = np.bincount(
        firms.sector, weights=np.maximum(state.revenue, 0.0), minlength=n_sectors
    )
    signal = np.divide(
        sector_fines, sector_rev, out=np.zeros_like(sector_fines), where=sector_rev > 0
    )
    publicity = const.publicity_decay * state.publicity + (1 - const.publicity_decay) * (
        signal / max(const.fine_rate, 1e-9)  # normalize to ~[0, 1]
    )

    interim = replace(state, y=y_new, fines=fines, audited=audited, publicity=publicity)
    trust, spend, revenue, cs, spend_utility = step_consumers(
        interim, firms, segments, graphs, theta, const, rng
    )
    alive, rev_hist, below = step_market_and_exit(interim, firms, theta, const, kappa, revenue, rng)
    # Newly exited firms leave this quarter's reported market immediately. Re-run
    # only the deterministic allocation with the same utility shock so HHI, CS,
    # spend, and revenue all describe the same survivor set.
    if not np.array_equal(alive, state.alive):
        spend, revenue, cs = allocate_spend(spend_utility, alive, graphs.market_mask, segments)
        rev_hist[:, -1] = revenue
    tenure = np.where((y_new > 0.5) & alive, state.tenure + 1, 0.0)

    new_state = WorldState(
        y=y_new * alive,
        alive=alive,
        revenue=revenue,
        tenure=tenure,
        fines=fines,
        audited=audited,
        spend=spend,
        trust=trust,
        publicity=publicity,
        rev_hist=rev_hist,
        below_floor=below,
        quarter=t + 1,
    )

    alive_f = alive.astype(np.float64)
    n_alive = max(float(alive_f.sum()), 1.0)
    rev_alive = revenue * alive_f
    terc = firms.size_tercile
    by_terc = tuple(
        float(np.sum(y_new * alive_f * (terc == k)) / max(np.sum(alive_f * (terc == k)), 1.0))
        for k in (0, 1, 2)
    )
    n_audits = int(audited.sum())
    outcome = QuarterOutcome(
        compliance_rate=float(np.sum(y_new * alive_f) / n_alive),
        compliance_rate_weighted=float(
            np.sum(y_new * rev_alive) / max(float(rev_alive.sum()), 1e-9)
        ),
        compliance_by_tercile=(by_terc[0], by_terc[1], by_terc[2]),
        hhi=hhi(revenue, alive),
        mean_trust=float(np.sum(segments.weight * trust) / segments.weight.sum()),
        consumer_surplus=cs,
        exit_rate_cum=float(1.0 - alive_f.sum() / firms.n),
        enforcement_cost=float(n_audits * const.audit_unit_cost),
        n_audits=n_audits,
    )
    covariates = {
        "perceived_risk": q,
        "cost_share": kappa,
        "neighbor_compliant_share": n_peer,
        "assoc_compliant_share": m_assoc,
        "privacy_rev_share": x_priv,
        "phase_phi": phi if isinstance(phi, np.ndarray) else np.full(firms.n, phi),
        "compliant_lag": state.y.copy(),
        "compliant": y_new,
        "audited": audited.astype(np.float64),
        "fined": (fines > 0).astype(np.float64),
        "alive": alive_f,
        "revenue": revenue,
        "segment_trust": trust,  # (S,) — the consumer survey samples this
    }
    return new_state, outcome, covariates


def regulator_reward(
    outcome: QuarterOutcome,
    baseline: QuarterOutcome,
    weights: tuple[float, float, float, float, float, float],
    const: Constants,
    n_firms: int,
) -> float:
    """r(t) (§7.6): a superset of the natural terms so no one number smuggles a value judgment."""
    w_c, w_h, w_s, w_e, w_t, w_x = weights
    e_max = const.audit_budget * n_firms * const.audit_unit_cost
    cs0 = abs(baseline.consumer_surplus) if baseline.consumer_surplus != 0 else 1.0
    return (
        w_c * outcome.compliance_rate
        - w_h * max(0.0, outcome.hhi - baseline.hhi) / 10_000.0
        - w_s * max(0.0, baseline.consumer_surplus - outcome.consumer_surplus) / cs0
        - w_e * outcome.enforcement_cost / max(e_max, 1e-9)
        + w_t * (outcome.mean_trust - baseline.mean_trust)
        - w_x * max(0.0, outcome.exit_rate_cum - baseline.exit_rate_cum)
    )


def backfire(outcome: QuarterOutcome, baseline: QuarterOutcome) -> bool:
    """The finding the client is paying for (§7.6): compliance up, HHI up, CS down."""
    return (
        outcome.compliance_rate > baseline.compliance_rate
        and outcome.hhi > baseline.hhi
        and outcome.consumer_surplus < baseline.consumer_surplus
    )


def initial_state(
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs: Graphs,
    const: Constants,
    rng: np.random.Generator,
) -> WorldState:
    """Quarter-0 state: nobody compliant, spend allocated by quality alone."""
    n_f, n_s = firms.n, segments.weight.size
    v = const.quality_weight * firms.quality[None, :] + rng.normal(
        0.0, const.spend_utility_noise, size=(n_s, n_f)
    )
    v_masked = np.where(graphs.market_mask, v, -np.inf)
    vmax = np.max(v_masked, axis=1, keepdims=True)
    expv = np.where(graphs.market_mask, np.exp(v - vmax), 0.0)
    shares = expv / np.maximum(expv.sum(axis=1, keepdims=True), 1e-12)
    spend = segments.budget[:, None] * shares
    revenue = spend.sum(axis=0)
    rev_hist = np.tile(revenue[:, None], (1, 3))
    return WorldState(
        y=np.zeros(n_f),
        alive=np.ones(n_f, dtype=bool),
        revenue=revenue,
        tenure=np.zeros(n_f),
        fines=np.zeros(n_f),
        audited=np.zeros(n_f, dtype=bool),
        spend=spend,
        trust=segments.trust0.copy(),
        publicity=np.zeros(int(firms.sector.max()) + 1),
        rev_hist=rev_hist,
        below_floor=np.zeros(n_f),
        quarter=0,
    )
