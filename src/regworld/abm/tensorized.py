"""Differentiable, population-tensor implementation of the regulation ABM.

The implementation intentionally uses plain PyTorch. AgentTorch 0.6 requires a
framework-specific registry/config/substep graph around the same tensor kernels,
while this stage needs a small callable that shares the Mesa model's public world
bundle. Supply and influence propagation use sparse matrix multiplication; hard
Bernoulli state transitions use straight-through gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import sparse

from regworld.rules import Constants, PolicyLevers, Theta
from regworld.types import RegWorldConfig

if TYPE_CHECKING:
    from regworld.abm.model import ObservedWorld


@dataclass
class TensorWorldState:
    """Torch counterpart of the shared quarterly world state."""

    y: torch.Tensor
    alive: torch.Tensor
    revenue: torch.Tensor
    tenure: torch.Tensor
    fines: torch.Tensor
    audited: torch.Tensor
    spend: torch.Tensor
    trust: torch.Tensor
    publicity: torch.Tensor
    rev_hist: torch.Tensor
    below_floor: torch.Tensor
    quarter: int


@dataclass(frozen=True)
class TensorQuarterOutcome:
    """Differentiable aggregate outcome tensors for one quarter."""

    compliance_rate: torch.Tensor
    compliance_rate_weighted: torch.Tensor
    compliance_by_tercile: torch.Tensor
    hhi: torch.Tensor
    mean_trust: torch.Tensor
    consumer_surplus: torch.Tensor
    exit_rate_cum: torch.Tensor
    enforcement_cost: torch.Tensor
    n_audits: torch.Tensor


@dataclass
class TensorTrajectory:
    """Tensorized analogue of the Mesa trajectory public contract."""

    outcomes: list[TensorQuarterOutcome]
    covariates: list[dict[str, torch.Tensor]]
    final_state: TensorWorldState
    compliance_probabilities: torch.Tensor

    def outcome_matrix(self) -> torch.Tensor:
        """Return ``(quarter, 7)`` differentiable aggregate outcomes."""
        if not self.outcomes:
            return torch.empty(
                (0, 7),
                device=self.final_state.y.device,
                dtype=self.final_state.y.dtype,
            )
        return torch.stack(
            [
                torch.stack(
                    [
                        outcome.compliance_rate,
                        outcome.compliance_rate_weighted,
                        outcome.hhi,
                        outcome.mean_trust,
                        outcome.consumer_surplus,
                        outcome.exit_rate_cum,
                        outcome.enforcement_cost,
                    ]
                )
                for outcome in self.outcomes
            ]
        )


def scipy_to_torch_sparse(
    matrix: sparse.spmatrix, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a SciPy adjacency to a coalesced Torch COO tensor."""
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        device=device,
        check_invariants=False,
    ).coalesce()


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _straight_through_bernoulli(
    probability: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    draws = torch.rand(probability.shape, generator=generator, device=probability.device)
    hard = draws < probability
    hard_float = hard.to(probability.dtype)
    return hard_float + probability - probability.detach(), hard_float


def _allocate_spend(
    utility: torch.Tensor,
    alive: torch.Tensor,
    market_mask: torch.Tensor,
    segment_weight: torch.Tensor,
    segment_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = market_mask & (alive.detach() > 0.5).unsqueeze(0)
    neg_inf = torch.full_like(utility, -torch.inf)
    masked = torch.where(mask, utility, neg_inf)
    inclusive_value = torch.logsumexp(masked, dim=1)
    inclusive_value = torch.where(
        torch.isfinite(inclusive_value), inclusive_value, torch.zeros_like(inclusive_value)
    )
    consumer_surplus = torch.sum(segment_weight * inclusive_value)
    vmax = torch.max(masked, dim=1, keepdim=True).values
    vmax = torch.where(torch.isfinite(vmax), vmax, torch.zeros_like(vmax))
    exp_utility = torch.where(mask, torch.exp(utility - vmax), torch.zeros_like(utility))
    shares = exp_utility / torch.clamp(exp_utility.sum(dim=1, keepdim=True), min=1e-12)
    spend = segment_budget.unsqueeze(1) * shares
    return spend, spend.sum(dim=0), consumer_surplus


def _association_share(
    y: torch.Tensor,
    alive: torch.Tensor,
    association: torch.Tensor,
    n_associations: int,
) -> torch.Tensor:
    alive_y = y * alive
    overall = alive_y.sum() / torch.clamp(alive.sum(), min=1.0)
    if n_associations == 0:
        return overall.expand_as(y)
    valid = association >= 0
    index = association[valid]
    sums = torch.zeros(n_associations, device=y.device, dtype=y.dtype)
    counts = torch.zeros_like(sums)
    sums = sums.scatter_add(0, index, alive_y[valid])
    counts = counts.scatter_add(0, index, alive[valid])
    means = sums / torch.clamp(counts, min=1.0)
    associated = means[association.clamp(min=0)]
    return torch.where(valid & (counts[association.clamp(min=0)] > 0), associated, overall)


def _clone_initial_state(
    state: Any, *, device: torch.device, dtype: torch.dtype
) -> TensorWorldState:
    def value(name: str) -> torch.Tensor:
        return _as_tensor(getattr(state, name), device=device, dtype=dtype).clone()

    return TensorWorldState(
        y=value("y"),
        alive=value("alive"),
        revenue=value("revenue"),
        tenure=value("tenure"),
        fines=value("fines"),
        audited=value("audited"),
        spend=value("spend"),
        trust=value("trust"),
        publicity=value("publicity"),
        rev_hist=value("rev_hist"),
        below_floor=value("below_floor"),
        quarter=int(state.quarter),
    )


def rollout_tensorized(
    cfg: RegWorldConfig,
    world: ObservedWorld,
    theta: Theta,
    policy: PolicyLevers,
    seed: int,
    quarters: int | None = None,
    treatment_start: np.ndarray | None = None,
) -> TensorTrajectory:
    """Roll out the observed-world ABM with sparse, differentiable Torch kernels.

    ``theta`` and ``policy`` fields may be scalar tensors with ``requires_grad``;
    Python floats are promoted on the configured device. The forward pass uses
    hard state transitions, while straight-through estimators retain gradients of
    compliance and exit probabilities.
    """
    device = torch.device(cfg.resolve_device())
    dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    firms = world.firms
    segments = world.segments
    graphs = world.graphs
    constants: Constants = getattr(world, "constants", Constants())
    state = _clone_initial_state(world.initial_state, device=device, dtype=dtype)
    n_quarters = cfg.horizon_quarters if quarters is None else quarters
    if n_quarters < 0:
        raise ValueError("quarters must be non-negative")

    supply = scipy_to_torch_sparse(graphs.supply_und, device=device, dtype=dtype)
    influence = scipy_to_torch_sparse(graphs.influence, device=device, dtype=dtype)
    market_mask = torch.as_tensor(graphs.market_mask, device=device, dtype=torch.bool)
    if treatment_start is None:
        treatment_start_tensor = torch.zeros(firms.n, device=device, dtype=dtype)
    else:
        if np.asarray(treatment_start).shape != (firms.n,):
            raise ValueError("treatment_start must have one zero-based onset per firm")
        treatment_start_tensor = _as_tensor(treatment_start, device=device, dtype=dtype)

    size = _as_tensor(firms.size, device=device, dtype=dtype)
    sector = torch.as_tensor(firms.sector, device=device, dtype=torch.long)
    data_intensity = _as_tensor(firms.data_intensity, device=device, dtype=dtype)
    cost_coef = _as_tensor(firms.cost_coef, device=device, dtype=dtype)
    quality = _as_tensor(firms.quality, device=device, dtype=dtype)
    base_margin = _as_tensor(firms.base_margin, device=device, dtype=dtype)
    association = torch.as_tensor(firms.association, device=device, dtype=torch.long)
    size_tercile = torch.as_tensor(firms.size_tercile, device=device, dtype=torch.long)
    segment_weight = _as_tensor(segments.weight, device=device, dtype=dtype)
    segment_privacy = _as_tensor(segments.privacy, device=device, dtype=dtype)
    segment_budget = _as_tensor(segments.budget, device=device, dtype=dtype)

    def parameter(name: str) -> torch.Tensor:
        return _as_tensor(getattr(theta, name), device=device, dtype=dtype)

    def lever(name: str) -> torch.Tensor:
        return _as_tensor(getattr(policy, name), device=device, dtype=dtype)

    enforcement = lever("enforcement")
    targeting = lever("targeting")
    phase_speed = lever("phase_speed")
    subsidy = lever("subsidy")

    outcomes: list[TensorQuarterOutcome] = []
    covariates: list[dict[str, torch.Tensor]] = []
    probabilities: list[torch.Tensor] = []
    n_associations = max(int(association.max().item()) + 1, 0)
    n_sectors = state.publicity.numel()
    for _ in range(n_quarters):
        phase_length = 12.0 - 10.0 * phase_speed
        time_since_start = state.quarter - treatment_start_tensor + 1.0
        phi = torch.where(
            time_since_start > 0,
            torch.clamp(
                time_since_start / phase_length,
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(time_since_start),
        )
        active = (treatment_start_tensor <= state.quarter).to(dtype)

        tau = torch.abs(targeting)
        large_target = size.pow(constants.target_gamma)
        small_target = torch.clamp(size, min=1e-9).reciprocal().pow(constants.target_gamma)
        size_target = torch.where(targeting >= 0, large_target, small_target)
        audit_weight = (1.0 - tau) + tau * size_target * (1.0 - state.y)
        audit_weight = audit_weight * state.alive * active
        audit_total = audit_weight.sum()
        expected_audits = enforcement * constants.audit_budget * size.numel()
        alpha = torch.where(
            audit_total > 0,
            torch.clamp(expected_audits * audit_weight / torch.clamp(audit_total, min=1e-12), 0, 1),
            torch.zeros_like(audit_weight),
        )

        publicity_by_firm = state.publicity[sector]
        perceived_risk = (
            alpha
            * constants.risk_scale
            * size.pow(constants.penalty_psi)
            * phi
            * (1.0 + constants.publicity_omega * publicity_by_firm)
            * (1.0 + constants.own_audit_boost * state.audited)
        )
        cost_share = (
            constants.cost_scale
            * cost_coef
            * data_intensity
            * size.pow(-parameter("gamma_scale"))
            * phi
            * (1.0 - parameter("ell_learn") * torch.clamp(state.tenure / 12.0, 0, 1))
        )
        cost_share = cost_share * (1.0 - subsidy * (size_tercile == 0).to(dtype))

        neighbor_num = torch.sparse.mm(supply, (state.y * state.alive).unsqueeze(1)).squeeze(1)
        neighbor_den = torch.sparse.mm(supply, state.alive.unsqueeze(1)).squeeze(1)
        neighbor_share = torch.where(
            neighbor_den > 0,
            neighbor_num / torch.clamp(neighbor_den, min=1e-12),
            torch.zeros_like(neighbor_num),
        )
        assoc_share = _association_share(
            state.y,
            state.alive,
            association,
            n_associations,
        )
        privacy_num = state.spend.transpose(0, 1) @ segment_privacy
        privacy_share = torch.where(
            state.revenue > 0,
            privacy_num / torch.clamp(state.revenue, min=1e-12),
            torch.zeros_like(privacy_num),
        )

        utility = (
            parameter("beta_0")
            + parameter("beta_enforce") * perceived_risk
            - parameter("beta_cost") * cost_share
            + parameter("beta_peer") * neighbor_share
            + parameter("beta_assoc") * assoc_share
            + parameter("beta_size") * torch.log(torch.clamp(size, min=1e-9))
            + parameter("beta_customer") * privacy_share
            + parameter("phi_phase") * phi
        )
        if cfg.behavior.sticky:
            utility = utility - parameter("beta_stick") * (1.0 - state.y)
        compliance_probability = torch.sigmoid(utility)
        y_new, y_hard = _straight_through_bernoulli(compliance_probability, generator)
        if cfg.behavior.attention < 1.0:
            reconsider = (
                torch.rand(size.shape, generator=generator, device=device) < cfg.behavior.attention
            ).to(dtype)
            y_new = reconsider * y_new + (1.0 - reconsider) * state.y
            y_hard = reconsider * y_hard + (1.0 - reconsider) * state.y.detach()
        y_new = y_new * state.alive
        y_hard = y_hard * (state.alive.detach() > 0.5).to(dtype)
        probabilities.append(compliance_probability)

        n_audits = int(torch.floor(alpha.detach().sum()).item())
        audited = torch.zeros_like(state.audited)
        eligible = torch.nonzero(alpha.detach() > 0, as_tuple=False).squeeze(1)
        if n_audits > 0 and eligible.numel() > 0:
            n_audits = min(n_audits, int(eligible.numel()))
            chosen_local = torch.multinomial(
                alpha.detach()[eligible], n_audits, replacement=False, generator=generator
            )
            audited[eligible[chosen_local]] = 1.0
        caught = (audited > 0.5) & (y_hard < 0.5) & (state.alive.detach() > 0.5)
        fine_fraction = min(constants.fine_rate, constants.fine_cap)
        fines = torch.where(
            caught,
            fine_fraction * torch.clamp(state.revenue, min=0.0),
            torch.zeros_like(state.revenue),
        )

        sector_fines = torch.zeros(n_sectors, device=device, dtype=dtype)
        sector_revenue = torch.zeros_like(sector_fines)
        sector_fines = sector_fines.scatter_add(0, sector, fines)
        sector_revenue = sector_revenue.scatter_add(0, sector, torch.clamp(state.revenue, min=0.0))
        signal = torch.where(
            sector_revenue > 0,
            sector_fines / torch.clamp(sector_revenue, min=1e-12),
            torch.zeros_like(sector_fines),
        )
        publicity = constants.publicity_decay * state.publicity + (
            1.0 - constants.publicity_decay
        ) * signal / max(constants.fine_rate, 1e-9)

        spend_share = state.spend / torch.clamp(state.spend.sum(dim=1, keepdim=True), min=1e-9)
        exposure = spend_share @ (y_new * state.alive)
        social = torch.sparse.mm(influence, state.trust.unsqueeze(1)).squeeze(1) - state.trust
        trust_noise = (
            torch.randn(state.trust.shape, generator=generator, device=device, dtype=dtype)
            * constants.trust_noise
        )
        trust = torch.clamp(
            state.trust
            + parameter("alpha_trust") * (exposure - state.trust)
            + parameter("rho_influence") * social
            + trust_noise,
            0,
            1,
        )
        spend_noise = (
            torch.randn(
                (segment_weight.numel(), size.numel()),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * constants.spend_utility_noise
        )
        spend_utility = (
            constants.quality_weight * quality.unsqueeze(0)
            + parameter("mu_privacy")
            * segment_privacy.unsqueeze(1)
            * (y_new * state.alive).unsqueeze(0)
            + spend_noise
        )
        spend, revenue, consumer_surplus = _allocate_spend(
            spend_utility,
            state.alive,
            market_mask,
            segment_weight,
            segment_budget,
        )

        margin = base_margin - y_new * cost_share - fines / torch.clamp(revenue, min=1e-9)
        rev_hist = torch.roll(state.rev_hist, -1, dims=1)
        rev_hist = torch.cat([rev_hist[:, :-1], revenue.unsqueeze(1)], dim=1)
        filled = min(state.quarter + 1, 3)
        rolling_revenue = rev_hist[:, -filled:].mean(dim=1)
        below_floor = torch.where(
            rolling_revenue < constants.exit_floor * size,
            state.below_floor + 1.0,
            torch.zeros_like(state.below_floor),
        )
        exit_probability = torch.clamp(
            parameter("delta_exit") * torch.abs(margin) / torch.clamp(size, min=1e-9),
            max=1.0,
        )
        at_risk = (state.alive.detach() > 0.5) & (below_floor.detach() >= 2)
        exit_draw, _ = _straight_through_bernoulli(exit_probability, generator)
        alive = state.alive * (1.0 - at_risk.to(dtype) * exit_draw)

        # Use the same utility shocks after exits, matching the shared rule.
        spend, revenue, consumer_surplus = _allocate_spend(
            spend_utility,
            alive,
            market_mask,
            segment_weight,
            segment_budget,
        )
        rev_hist = torch.cat([rev_hist[:, :-1], revenue.unsqueeze(1)], dim=1)
        tenure = torch.where(
            (y_hard > 0.5) & (alive.detach() > 0.5),
            state.tenure + 1.0,
            torch.zeros_like(state.tenure),
        )

        alive_count = torch.clamp(alive.sum(), min=1.0)
        revenue_alive = revenue * alive
        by_tercile = torch.stack(
            [
                torch.sum(y_new * alive * (size_tercile == tercile).to(dtype))
                / torch.clamp(torch.sum(alive * (size_tercile == tercile).to(dtype)), min=1.0)
                for tercile in (0, 1, 2)
            ]
        )
        revenue_total = torch.clamp(revenue_alive.sum(), min=1e-9)
        revenue_shares = revenue_alive / revenue_total
        outcome = TensorQuarterOutcome(
            compliance_rate=torch.sum(y_new * alive) / alive_count,
            compliance_rate_weighted=torch.sum(y_new * revenue_alive) / revenue_total,
            compliance_by_tercile=by_tercile,
            hhi=10_000.0 * torch.sum(revenue_shares.square()),
            mean_trust=torch.sum(segment_weight * trust) / segment_weight.sum(),
            consumer_surplus=consumer_surplus,
            exit_rate_cum=1.0 - alive.sum() / size.numel(),
            enforcement_cost=audited.sum() * constants.audit_unit_cost,
            n_audits=audited.sum().to(torch.long),
        )
        covariates.append(
            {
                "perceived_risk": perceived_risk,
                "cost_share": cost_share,
                "neighbor_compliant_share": neighbor_share,
                "assoc_compliant_share": assoc_share,
                "privacy_rev_share": privacy_share,
                "phase_phi": phi,
                "compliant_lag": state.y,
                "compliant_probability": compliance_probability,
                "compliant": y_new,
                "audited": audited,
                "fined": (fines > 0).to(dtype),
                "alive": alive,
                "revenue": revenue,
                "segment_trust": trust,
            }
        )
        state = replace(
            state,
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
            below_floor=below_floor,
            quarter=state.quarter + 1,
        )
        outcomes.append(outcome)

    probability_matrix = (
        torch.stack(probabilities)
        if probabilities
        else torch.empty((0, size.numel()), device=device)
    )
    return TensorTrajectory(
        outcomes=outcomes,
        covariates=covariates,
        final_state=state,
        compliance_probabilities=probability_matrix,
    )
