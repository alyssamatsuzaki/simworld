"""Mesa 3 regulation model estimated from the observed Stage-1 artifacts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import mesa
import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit

from regworld import rules
from regworld.abm.agents import AssociationAgent, FirmAgent, RegulatorAgent, SegmentAgent
from regworld.abm.policies import levers_from_config
from regworld.data import store
from regworld.data.schema import EDGES, validate_table
from regworld.types import RegWorldConfig


@dataclass(frozen=True)
class ObservedWorld:
    """All inputs needed by the estimated simulator, reconstructed from observables."""

    firms: rules.FirmAttributes
    segments: rules.SegmentAttributes
    graphs: rules.Graphs
    initial_state: rules.WorldState
    theta: rules.Theta


@dataclass(frozen=True)
class StrategicControls:
    """Vectorized pre-draw controls used by the optional strategic-firm environment."""

    utility_bonus: np.ndarray
    detection_multiplier: np.ndarray
    association_enforcement_multiplier: np.ndarray
    action_cost: np.ndarray

    @classmethod
    def neutral(cls, n_firms: int, n_associations: int) -> StrategicControls:
        return cls(
            utility_bonus=np.zeros(n_firms, dtype=np.float64),
            detection_multiplier=np.ones(n_firms, dtype=np.float64),
            association_enforcement_multiplier=np.ones(n_associations, dtype=np.float64),
            action_cost=np.zeros(n_firms, dtype=np.float64),
        )


@dataclass(frozen=True)
class Trajectory:
    """ABM-owned trajectory contract; independent of the world-builder package."""

    outcomes: tuple[rules.QuarterOutcome, ...]
    aggregate: pl.DataFrame
    firm_panel: pl.DataFrame
    events: tuple[dict[str, Any], ...]
    final_state: rules.WorldState


def _read_edges(cfg: RegWorldConfig, name: str) -> pl.DataFrame:
    path = store.observed_dir(cfg) / "graphs" / f"{name}.parquet"
    frame = pl.read_parquet(path)
    validate_table(frame, EDGES)
    return frame


def _numeric_id(value: object, prefix: str = "") -> int:
    text = str(value)
    if prefix and text.startswith(prefix):
        text = text.removeprefix(prefix)
    return int(text)


def _observed_graphs(cfg: RegWorldConfig, n_firms: int, n_segments: int) -> rules.Graphs:
    supply = _read_edges(cfg, "supply_edges")
    supply_pairs: set[tuple[int, int]] = set()
    for src, dst in supply.iter_rows():
        i, j = _numeric_id(src, "firm_"), _numeric_id(dst, "firm_")
        if 0 <= i < n_firms and 0 <= j < n_firms and i != j:
            supply_pairs.add((i, j))
            supply_pairs.add((j, i))
    if supply_pairs:
        row, col = zip(*sorted(supply_pairs), strict=True)
        supply_und = sparse.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(n_firms, n_firms), dtype=np.float64
        )
    else:
        supply_und = sparse.csr_matrix((n_firms, n_firms), dtype=np.float64)

    influence_frame = _read_edges(cfg, "influence_edges")
    influence_pairs: set[tuple[int, int]] = set()
    for src, dst in influence_frame.iter_rows():
        i, j = _numeric_id(src, "seg_"), _numeric_id(dst, "seg_")
        if 0 <= i < n_segments and 0 <= j < n_segments and i != j:
            influence_pairs.add((i, j))
            influence_pairs.add((j, i))
    if influence_pairs:
        row, col = zip(*sorted(influence_pairs), strict=True)
        influence = sparse.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(n_segments, n_segments), dtype=np.float64
        )
    else:
        influence = sparse.csr_matrix((n_segments, n_segments), dtype=np.float64)
    degree = np.asarray(influence.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree == 0.0)
    if isolated.size:
        influence = influence.tolil()
        influence[isolated, isolated] = 1.0
        influence = influence.tocsr()
        degree = np.asarray(influence.sum(axis=1)).ravel()
    influence = sparse.diags(1.0 / np.maximum(degree, 1.0)) @ influence

    market_mask = np.zeros((n_segments, n_firms), dtype=bool)
    for src, dst in _read_edges(cfg, "market_edges").iter_rows():
        src_text, dst_text = str(src), str(dst)
        if src_text.startswith("seg_"):
            segment_id = _numeric_id(src_text, "seg_")
            firm_id = _numeric_id(dst_text, "firm_")
        else:
            segment_id = _numeric_id(dst_text, "seg_")
            firm_id = _numeric_id(src_text, "firm_")
        if 0 <= segment_id < n_segments and 0 <= firm_id < n_firms:
            market_mask[segment_id, firm_id] = True
    for isolated_segment in np.flatnonzero(~market_mask.any(axis=1)):
        market_mask[isolated_segment, :] = True
    return rules.Graphs(
        supply_und=supply_und,
        influence=influence.tocsr(),
        market_mask=market_mask,
    )


def _observed_segments(
    cfg: RegWorldConfig, survey: pl.DataFrame, total_budget: float
) -> rules.SegmentAttributes:
    n_segments = cfg.population.n_consumer_segments
    counts = np.ones(n_segments, dtype=np.float64)
    privacy = np.full(n_segments, 0.5, dtype=np.float64)
    trust0 = np.full(n_segments, 0.5, dtype=np.float64)
    if not survey.is_empty():
        summary = survey.group_by("segment_id").agg(
            pl.len().alias("n"),
            (pl.col("privacy_bucket").cast(pl.Float64).mean() / 2.0).alias("privacy"),
        )
        latest = survey.sort("quarter").group_by("segment_id").last()
        latest_trust = dict(
            zip(latest["segment_id"].to_list(), latest["trust_reported"].to_list(), strict=True)
        )
        for row in summary.iter_rows(named=True):
            segment_id = int(row["segment_id"])
            if 0 <= segment_id < n_segments:
                counts[segment_id] += float(row["n"])
                privacy[segment_id] = float(row["privacy"])
                trust0[segment_id] = float(latest_trust[segment_id])
    weight = counts / counts.sum()
    return rules.SegmentAttributes(
        weight=weight,
        privacy=np.clip(privacy, 0.0, 1.0),
        budget=weight * total_budget,
        trust0=np.clip(trust0, 0.0, 1.0),
    )


def load_observed_world(cfg: RegWorldConfig, seed: int | None = None) -> ObservedWorld:
    """Reconstruct a forecast world strictly from observed Parquet inputs."""
    del seed  # initial forecast state is deterministic; the model seed drives dynamics
    registry = store.read_observed(cfg, "firm_registry").sort("firm_id")
    panel = store.read_observed(cfg, "firm_panel")
    survey = store.read_observed(cfg, "consumer_survey")
    firm_ids = registry["firm_id"].to_numpy()
    n_firms = registry.height
    if not np.array_equal(firm_ids, np.arange(n_firms)):
        raise ValueError("observed firm ids must be contiguous integers starting at zero")

    size_decile = registry["size_decile"].to_numpy().astype(np.int64)
    log_size = np.log(size_decile.astype(np.float64) + 1.0)
    log_size -= float(np.median(log_size))
    size = np.exp(log_size)
    sector = registry["sector"].to_numpy().astype(np.int64)
    data_intensity = registry["data_intensity"].to_numpy().astype(np.float64)
    cost_coef = np.maximum(registry["cost_index"].to_numpy().astype(np.float64), 0.01)
    association = registry["association"].to_numpy().astype(np.int64)
    size_tercile = np.where(size_decile <= 2, 0, np.where(size_decile <= 5, 1, 2)).astype(np.int64)

    latest = panel.sort("quarter").group_by("firm_id").last()
    latest_by_id = {int(row["firm_id"]): row for row in latest.iter_rows(named=True)}
    y = np.zeros(n_firms, dtype=np.float64)
    alive = np.ones(n_firms, dtype=bool)
    audited = np.zeros(n_firms, dtype=bool)
    region = np.zeros(n_firms, dtype=np.int64)
    revenue_signal = size.copy()
    sampled_revenue = latest["revenue_noisy"].to_numpy().astype(np.float64)
    revenue_scale = float(np.median(sampled_revenue[sampled_revenue > 0.0])) or 1.0
    for firm_id in range(n_firms):
        row = latest_by_id.get(firm_id)
        if row is None:
            continue
        region[firm_id] = int(row["region"])
        revenue_signal[firm_id] = max(float(row["revenue_noisy"]) / revenue_scale, 1e-6)

    quality_raw = np.log1p(revenue_signal)
    quality_sd = max(float(quality_raw.std()), 1e-6)
    quality = (quality_raw - float(quality_raw.mean())) / quality_sd
    base_margin = 0.30 + 0.20 * data_intensity
    firms = rules.FirmAttributes(
        size=size,
        sector=sector,
        data_intensity=data_intensity,
        cost_coef=cost_coef,
        quality=quality,
        base_margin=base_margin,
        z=np.zeros(n_firms, dtype=np.float64),
        association=association,
        region=region,
        size_tercile=size_tercile,
    )
    segments = _observed_segments(cfg, survey, total_budget=float(size.sum()))
    graphs = _observed_graphs(cfg, n_firms, segments.weight.size)
    constants = rules.Constants()
    utility = constants.quality_weight * quality[None, :]
    spend, revenue, _ = rules.allocate_spend(utility, alive, graphs.market_mask, segments)

    publicity = np.zeros(int(sector.max()) + 1, dtype=np.float64)
    initial_state = rules.WorldState(
        y=y,
        alive=alive,
        revenue=revenue,
        tenure=np.zeros(n_firms, dtype=np.float64),
        fines=np.zeros(n_firms, dtype=np.float64),
        audited=audited,
        spend=spend,
        trust=segments.trust0.copy(),
        publicity=publicity,
        rev_hist=np.tile(revenue[:, None], (1, 3)),
        below_floor=np.zeros(n_firms, dtype=np.float64),
        quarter=0,
    )
    return ObservedWorld(
        firms=firms,
        segments=segments,
        graphs=graphs,
        initial_state=initial_state,
        theta=rules.Theta(beta_capacity=0.0),
    )


def strategic_controls_from_actions(
    actions: Mapping[int, np.ndarray], firms: rules.FirmAttributes, revenue: np.ndarray
) -> StrategicControls:
    """Map `[invest, lobby, evade]` actions to the vector hook used by `step`."""
    n_associations = max(int(firms.association.max()) + 1, 0)
    controls = StrategicControls.neutral(firms.n, n_associations)
    utility_bonus = controls.utility_bonus.copy()
    detection = controls.detection_multiplier.copy()
    association_multiplier = controls.association_enforcement_multiplier.copy()
    action_cost = controls.action_cost.copy()
    for firm_id, raw_action in actions.items():
        if not 0 <= firm_id < firms.n:
            raise IndexError(f"strategic firm id {firm_id} outside [0, {firms.n})")
        action = np.asarray(raw_action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("each strategic action must be a finite shape-(3,) array")
        invest, lobby, evade = np.clip(action, 0.0, 1.0)
        utility_bonus[firm_id] = 2.0 * invest
        detection[firm_id] = 1.0 - 0.8 * evade
        association_id = int(firms.association[firm_id])
        if 0 <= association_id < association_multiplier.size:
            association_multiplier[association_id] *= 1.0 - 0.5 * lobby
        action_cost[firm_id] = max(float(revenue[firm_id]), 0.0) * (
            0.03 * invest + 0.02 * lobby + 0.02 * evade
        )
    return StrategicControls(utility_bonus, detection, association_multiplier, action_cost)


class RegulationModel(mesa.Model):
    """Vectorized Mesa model whose agents are inspectable views of array state."""

    def __init__(
        self,
        cfg: RegWorldConfig,
        world: ObservedWorld | None = None,
        theta: rules.Theta | None = None,
        policy: rules.PolicyLevers | None = None,
        seed: int | None = None,
    ) -> None:
        self.seed_value = cfg.seed if seed is None else seed
        super().__init__(rng=self.seed_value)
        self.cfg = cfg
        self.world = world or load_observed_world(cfg, self.seed_value)
        self.firms = self.world.firms
        self.segments = self.world.segments
        self.graphs = self.world.graphs
        self.theta = theta or self.world.theta
        self.policy = policy or levers_from_config(cfg.policy)
        self.constants = rules.Constants()
        self.np_rng = np.random.default_rng(self.seed_value)
        self.state = copy.deepcopy(self.world.initial_state)
        self.last_outcome: rules.QuarterOutcome | None = None
        self.baseline_outcome = self._snapshot_outcome(self.state)
        self.last_covariates: dict[str, np.ndarray] = {}
        self.last_firm_rewards = np.zeros(self.firms.n, dtype=np.float64)
        self.last_regulator_reward = 0.0
        self.last_strategic_controls = StrategicControls.neutral(
            self.firms.n, max(int(self.firms.association.max()) + 1, 0)
        )
        self.outcomes: list[rules.QuarterOutcome] = []
        self.events: list[dict[str, Any]] = []
        self._records: list[dict[str, float | int | bool]] = []
        self._pending_step: (
            tuple[
                rules.PolicyLevers | None,
                Mapping[int, np.ndarray] | None,
                StrategicControls | None,
            ]
            | None
        ) = None

        for firm_id in range(self.firms.n):
            FirmAgent(self, firm_id)
        for segment_id in range(self.segments.weight.size):
            SegmentAgent(self, segment_id)
        for association_id in range(
            self.last_strategic_controls.association_enforcement_multiplier.size
        ):
            AssociationAgent(self, association_id)
        RegulatorAgent(self)

        from regworld.abm.collect import make_data_collector

        self.datacollector = make_data_collector()

    @property
    def quarter(self) -> int:
        return self.state.quarter

    @property
    def compliance_rate(self) -> float:
        return self.last_outcome.compliance_rate if self.last_outcome else 0.0

    @property
    def compliance_rate_weighted(self) -> float:
        return self.last_outcome.compliance_rate_weighted if self.last_outcome else 0.0

    @property
    def compliance_small(self) -> float:
        return self.last_outcome.compliance_by_tercile[0] if self.last_outcome else 0.0

    @property
    def compliance_mid(self) -> float:
        return self.last_outcome.compliance_by_tercile[1] if self.last_outcome else 0.0

    @property
    def compliance_large(self) -> float:
        return self.last_outcome.compliance_by_tercile[2] if self.last_outcome else 0.0

    @property
    def hhi(self) -> float:
        return self.last_outcome.hhi if self.last_outcome else 0.0

    @property
    def mean_trust(self) -> float:
        return self.last_outcome.mean_trust if self.last_outcome else 0.0

    @property
    def consumer_surplus(self) -> float:
        return self.last_outcome.consumer_surplus if self.last_outcome else 0.0

    @property
    def exit_rate(self) -> float:
        return self.last_outcome.exit_rate_cum if self.last_outcome else 0.0

    @property
    def enforcement_cost(self) -> float:
        return self.last_outcome.enforcement_cost if self.last_outcome else 0.0

    @property
    def n_audits(self) -> int:
        return self.last_outcome.n_audits if self.last_outcome else 0

    @property
    def reward(self) -> float:
        return self.last_regulator_reward

    @property
    def backfire(self) -> bool:
        return bool(self.last_outcome and rules.backfire(self.last_outcome, self.baseline_outcome))

    def _snapshot_outcome(self, state: rules.WorldState) -> rules.QuarterOutcome:
        alive_f = state.alive.astype(np.float64)
        n_alive = max(float(alive_f.sum()), 1.0)
        rev_alive = state.revenue * alive_f
        terc = self.firms.size_tercile
        by_tercile = tuple(
            float(np.sum(state.y * alive_f * (terc == k)) / max(np.sum(alive_f * (terc == k)), 1.0))
            for k in (0, 1, 2)
        )
        utility = self.constants.quality_weight * self.firms.quality[None, :]
        _, _, consumer_surplus = rules.allocate_spend(
            utility, state.alive, self.graphs.market_mask, self.segments
        )
        return rules.QuarterOutcome(
            compliance_rate=float(np.sum(state.y * alive_f) / n_alive),
            compliance_rate_weighted=float(
                np.sum(state.y * rev_alive) / max(float(rev_alive.sum()), 1e-9)
            ),
            compliance_by_tercile=(by_tercile[0], by_tercile[1], by_tercile[2]),
            hhi=rules.hhi(state.revenue, state.alive),
            mean_trust=float(np.sum(self.segments.weight * state.trust)),
            consumer_surplus=consumer_surplus,
            exit_rate_cum=float(1.0 - alive_f.sum() / self.firms.n),
            enforcement_cost=0.0,
            n_audits=0,
        )

    def _validated_controls(self, controls: StrategicControls) -> StrategicControls:
        n_associations = self.last_strategic_controls.association_enforcement_multiplier.size
        expected = {
            "utility_bonus": (controls.utility_bonus, (self.firms.n,)),
            "detection_multiplier": (controls.detection_multiplier, (self.firms.n,)),
            "association_enforcement_multiplier": (
                controls.association_enforcement_multiplier,
                (n_associations,),
            ),
            "action_cost": (controls.action_cost, (self.firms.n,)),
        }
        for name, (values, shape) in expected.items():
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite array with shape {shape}")
        if (
            np.any((controls.detection_multiplier < 0.0) | (controls.detection_multiplier > 1.0))
            or np.any(
                (controls.association_enforcement_multiplier < 0.0)
                | (controls.association_enforcement_multiplier > 1.0)
            )
            or np.any(controls.action_cost < 0.0)
        ):
            raise ValueError("strategic multipliers must be in [0,1] and costs nonnegative")
        return controls

    def step(self) -> None:
        """Mesa-scheduled zero-argument step using default or queued controls."""
        pending = self._pending_step
        self._pending_step = None
        if pending is None:
            self._step_impl()
        else:
            self._step_impl(*pending)

    def step_with_controls(
        self,
        policy: rules.PolicyLevers | None = None,
        strategic_actions: Mapping[int, np.ndarray] | None = None,
        controls: StrategicControls | None = None,
    ) -> None:
        """Queue one controlled transition and advance through Mesa's step wrapper."""
        if self._pending_step is not None:
            raise RuntimeError("a controlled step is already pending")
        self._pending_step = (policy, strategic_actions, controls)
        self.step()

    def _step_impl(
        self,
        policy: rules.PolicyLevers | None = None,
        strategic_actions: Mapping[int, np.ndarray] | None = None,
        controls: StrategicControls | None = None,
    ) -> None:
        if strategic_actions is not None and controls is not None:
            raise ValueError("pass strategic_actions or controls, not both")
        levers = policy or self.policy
        if strategic_actions is not None:
            controls = strategic_controls_from_actions(
                strategic_actions, self.firms, self.state.revenue
            )
        if controls is None:
            controls = StrategicControls.neutral(
                self.firms.n, self.last_strategic_controls.association_enforcement_multiplier.size
            )
        controls = self._validated_controls(controls)
        self.last_strategic_controls = controls
        previous = self.state
        t = previous.quarter
        active = np.ones(self.firms.n, dtype=np.float64)
        phi = rules.phase_progress(t, levers, 0)
        if np.ndim(phi) == 0:
            phi = np.full(self.firms.n, float(phi), dtype=np.float64)

        alpha = rules.audit_probabilities(previous, self.firms, self.constants, levers, active)
        associations = self.firms.association
        valid_association = (associations >= 0) & (
            associations < controls.association_enforcement_multiplier.size
        )
        firm_enforcement = np.ones(self.firms.n, dtype=np.float64)
        firm_enforcement[valid_association] = controls.association_enforcement_multiplier[
            associations[valid_association]
        ]
        alpha = np.clip(alpha * firm_enforcement, 0.0, 1.0)
        perceived_risk = rules.perceived_risk(alpha, self.firms, previous, self.constants, phi)
        cost_share = rules.compliance_cost_share(
            self.firms, previous, self.theta, self.constants, phi, levers
        )
        neighbour = rules.neighbour_share(
            previous.y, previous.alive.astype(np.float64), self.graphs
        )
        association = rules.association_share(
            previous.y,
            previous.alive,
            self.firms,
            max(int(self.firms.association.max()) + 1, 0),
        )
        privacy = rules.privacy_revenue_share(previous.spend, previous.revenue, self.segments)
        utility = rules.firm_utility(
            theta=self.theta,
            const=self.constants,
            q=perceived_risk,
            kappa=cost_share,
            n_peer=neighbour,
            m_assoc=association,
            log_size=np.log(np.maximum(self.firms.size, 1e-9)),
            x_privacy=privacy,
            phi=phi,
            y_prev=previous.y,
            z=None,
            sticky=self.cfg.behavior.sticky,
            interacted=False,
        )
        compliance_probability = expit(utility + controls.utility_bonus)
        compliant = (self.np_rng.random(self.firms.n) < compliance_probability).astype(np.float64)
        if self.cfg.behavior.attention < 1.0:
            reconsider = self.np_rng.random(self.firms.n) < self.cfg.behavior.attention
            compliant = np.where(reconsider, compliant, previous.y)
        compliant *= previous.alive

        audited = rules.draw_audits(alpha, self.np_rng)
        if np.all(controls.detection_multiplier == 1.0):
            detected = audited
        else:
            detected = audited & (self.np_rng.random(self.firms.n) < controls.detection_multiplier)
        caught = detected & (compliant < 0.5) & previous.alive
        fines = np.where(
            caught,
            min(self.constants.fine_rate, self.constants.fine_cap)
            * np.maximum(previous.revenue, 0.0),
            0.0,
        )
        n_sectors = int(self.firms.sector.max()) + 1
        sector_fines = np.bincount(self.firms.sector, weights=fines, minlength=n_sectors)
        sector_revenue = np.bincount(
            self.firms.sector, weights=np.maximum(previous.revenue, 0.0), minlength=n_sectors
        )
        fine_signal = np.divide(
            sector_fines,
            sector_revenue,
            out=np.zeros_like(sector_fines),
            where=sector_revenue > 0.0,
        )
        publicity = self.constants.publicity_decay * previous.publicity + (
            1.0 - self.constants.publicity_decay
        ) * (fine_signal / max(self.constants.fine_rate, 1e-9))

        interim = replace(previous, y=compliant, fines=fines, audited=audited, publicity=publicity)
        trust, spend, revenue, consumer_surplus, spend_utility = rules.step_consumers(
            interim,
            self.firms,
            self.segments,
            self.graphs,
            self.theta,
            self.constants,
            self.np_rng,
        )
        alive, rev_hist, below_floor = rules.step_market_and_exit(
            interim,
            self.firms,
            self.theta,
            self.constants,
            cost_share,
            revenue,
            self.np_rng,
        )
        if not np.array_equal(alive, previous.alive):
            spend, revenue, consumer_surplus = rules.allocate_spend(
                spend_utility, alive, self.graphs.market_mask, self.segments
            )
            rev_hist[:, -1] = revenue
        tenure = np.where((compliant > 0.5) & alive, previous.tenure + 1.0, 0.0)
        self.state = rules.WorldState(
            y=compliant * alive,
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
            quarter=t + 1,
        )

        alive_float = alive.astype(np.float64)
        n_alive = max(float(alive_float.sum()), 1.0)
        revenue_alive = revenue * alive_float
        by_tercile = tuple(
            float(
                np.sum(compliant * alive_float * (self.firms.size_tercile == k))
                / max(np.sum(alive_float * (self.firms.size_tercile == k)), 1.0)
            )
            for k in (0, 1, 2)
        )
        outcome = rules.QuarterOutcome(
            compliance_rate=float(np.sum(compliant * alive_float) / n_alive),
            compliance_rate_weighted=float(
                np.sum(compliant * revenue_alive) / max(float(revenue_alive.sum()), 1e-9)
            ),
            compliance_by_tercile=(by_tercile[0], by_tercile[1], by_tercile[2]),
            hhi=rules.hhi(revenue, alive),
            mean_trust=float(np.sum(self.segments.weight * trust)),
            consumer_surplus=consumer_surplus,
            exit_rate_cum=float(1.0 - alive_float.sum() / self.firms.n),
            enforcement_cost=float(audited.sum() * self.constants.audit_unit_cost),
            n_audits=int(audited.sum()),
        )
        self.last_outcome = outcome
        self.outcomes.append(outcome)
        weights = (
            self.cfg.objective.w_c,
            self.cfg.objective.w_h,
            self.cfg.objective.w_s,
            self.cfg.objective.w_e,
            self.cfg.objective.w_t,
            self.cfg.objective.w_x,
        )
        self.last_regulator_reward = rules.regulator_reward(
            outcome, self.baseline_outcome, weights, self.constants, self.firms.n
        )
        compliance_cost = compliant * cost_share * np.maximum(revenue, 0.0)
        self.last_firm_rewards = np.where(
            alive,
            self.firms.base_margin * revenue - compliance_cost - fines - controls.action_cost,
            0.0,
        )
        self.last_covariates = {
            "perceived_risk": perceived_risk,
            "cost_share": cost_share,
            "neighbor_compliant_share": neighbour,
            "assoc_compliant_share": association,
            "privacy_rev_share": privacy,
            "phase_phi": phi,
            "compliant_lag": previous.y.copy(),
            "compliance_probability": compliance_probability,
            "utility_bonus": controls.utility_bonus.copy(),
            "detection_multiplier": controls.detection_multiplier.copy(),
        }
        exited = np.flatnonzero(previous.alive & ~alive).astype(int).tolist()
        self.events.append(
            {
                "quarter": self.state.quarter,
                "audited_firm_ids": np.flatnonzero(audited).astype(int).tolist(),
                "fined_firm_ids": np.flatnonzero(fines > 0.0).astype(int).tolist(),
                "exited_firm_ids": exited,
            }
        )
        self._records.append(
            {
                "quarter": self.state.quarter,
                "compliance_rate": outcome.compliance_rate,
                "compliance_rate_weighted": outcome.compliance_rate_weighted,
                "compliance_small": outcome.compliance_by_tercile[0],
                "compliance_mid": outcome.compliance_by_tercile[1],
                "compliance_large": outcome.compliance_by_tercile[2],
                "hhi": outcome.hhi,
                "mean_trust": outcome.mean_trust,
                "consumer_surplus": outcome.consumer_surplus,
                "exit_rate": outcome.exit_rate_cum,
                "enforcement_cost": outcome.enforcement_cost,
                "n_audits": outcome.n_audits,
                "reward": self.last_regulator_reward,
                "backfire": rules.backfire(outcome, self.baseline_outcome),
            }
        )
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def run(
        self,
        quarters: int,
        policy: rules.PolicyLevers | None = None,
        strategic_actions_by_quarter: Mapping[int, Mapping[int, np.ndarray]] | None = None,
    ) -> Trajectory:
        if quarters < 0:
            raise ValueError("quarters must be nonnegative")
        for _ in range(quarters):
            actions = (
                strategic_actions_by_quarter.get(self.state.quarter)
                if strategic_actions_by_quarter is not None
                else None
            )
            if policy is None and actions is None:
                self.step()
            else:
                self.step_with_controls(policy=policy, strategic_actions=actions)
        from regworld.abm.collect import firm_panel_from_collector, model_frame_from_collector

        return Trajectory(
            outcomes=tuple(self.outcomes),
            aggregate=model_frame_from_collector(self),
            firm_panel=firm_panel_from_collector(self),
            events=tuple(self.events),
            final_state=copy.deepcopy(self.state),
        )
