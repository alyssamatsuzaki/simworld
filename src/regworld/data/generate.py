"""generate_ground_truth (§9): builds the world, runs Regime P, degrades it into
observed/, and seals the answer key into oracle/.

This module is a WORLD BUILDER: it may import regworld.dgp (see the firewall
allowlist in tests/test_no_dgp_leakage.py). Nothing downstream of Stage 1 may.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from scipy.special import expit, logsumexp

from regworld.abm.policies import STATIC_POLICIES
from regworld.data import store
from regworld.data.schema import EDGES
from regworld.dgp import history as dgp_history
from regworld.dgp import observation as obs
from regworld.dgp.dynamics import Trajectory, run_dgp
from regworld.dgp.world import (
    CONSTANTS,
    THETA_STAR,
    generate_firms,
    generate_segments,
    theta_star_dict,
)
from regworld.graphs.analyze import analyze_graphs
from regworld.graphs.build import RegGraphs, build_graphs, edges_frame
from regworld.rules import (
    OUTCOME_VARIABLES,
    FirmAttributes,
    Graphs,
    QuarterOutcome,
    SegmentAttributes,
    WorldState,
    backfire,
    firm_utility,
    hhi,
    initial_state,
    regulator_reward,
)
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

BALANCED_WEIGHTS = (1.0, 0.5, 0.5, 0.1, 0.3, 0.3)


@dataclass
class GenerationResult:
    observed_paths: list[Path]
    sealed_paths: list[Path]  # answer-key artifacts; named to keep the leakage grep strict


def _edges_df(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"src": [a for a, _ in pairs], "dst": [b for _, b in pairs]}).cast(
        {"src": pl.Utf8, "dst": pl.Utf8}
    )


def _write_graph_edges(cfg: RegWorldConfig, reg: RegGraphs) -> list[Path]:
    """Observed edge lists under observed/graphs; TRUE lists under oracle/true_graphs."""
    out: list[Path] = []
    gdir = store.observed_dir(cfg) / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    tdir = store.oracle_dir(cfg) / "true_graphs"
    tdir.mkdir(parents=True, exist_ok=True)
    observed = {
        "supply_edges": edges_frame(reg.supply_obs),
        "influence_edges": edges_frame(reg.influence_obs),
        "market_edges": edges_frame(reg.market),
        "membership_edges": edges_frame(reg.membership),
    }
    truth = {
        "supply_edges": edges_frame(reg.supply_true),
        "influence_edges": edges_frame(reg.influence_true),
        "market_edges": edges_frame(reg.market),
        "membership_edges": edges_frame(reg.membership),
    }
    for name, pairs in observed.items():
        df = _edges_df(pairs)
        from regworld.data.schema import validate_table

        validate_table(df, EDGES)
        p = gdir / f"{name}.parquet"
        df.write_parquet(p, compression="snappy")
        out.append(p)
    for name, pairs in truth.items():
        p = tdir / f"{name}.parquet"
        _edges_df(pairs).write_parquet(p, compression="snappy")
        out.append(p)
    return out


def _regime_f_truth(
    cfg: RegWorldConfig,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs_true: Graphs,
    initial_future_state: WorldState,
    baseline: QuarterOutcome,
) -> Path:
    """Ground-truth 24-quarter futures under every static policy x seed (§8)."""
    policies = list(STATIC_POLICIES)
    n_p, n_s, n_q = len(policies), len(cfg.seeds), cfg.horizon_quarters
    cube = np.zeros((n_p, n_s, n_q, len(OUTCOME_VARIABLES)))
    for ip, pname in enumerate(policies):
        levers = STATIC_POLICIES[pname]
        for isd, seed in enumerate(cfg.seeds):
            traj = run_dgp(
                cfg,
                firms,
                segments,
                graphs_true,
                levers,
                seed + 40_000,
                n_q,
                t_start=0,
                start_state=_copy_state(initial_future_state),
            )
            for t, o in enumerate(traj.outcomes):
                cube[ip, isd, t, :7] = [
                    o.compliance_rate,
                    o.compliance_rate_weighted,
                    o.hhi,
                    o.mean_trust,
                    o.consumer_surplus,
                    o.exit_rate_cum,
                    o.enforcement_cost,
                ]
                cube[ip, isd, t, 7] = regulator_reward(
                    o, baseline, BALANCED_WEIGHTS, CONSTANTS, firms.n
                )
                cube[ip, isd, t, 8] = float(backfire(o, baseline))
    ds = xr.Dataset(
        {"truth": (("policy", "seed", "quarter", "variable"), cube)},
        coords={
            "policy": policies,
            "seed": list(cfg.seeds),
            "quarter": np.arange(1, n_q + 1),
            "variable": OUTCOME_VARIABLES,
        },
    )
    path = store.oracle_dir(cfg) / "regime_f_truth.zarr"
    ds.to_zarr(path, mode="w")
    return path


def _copy_state(state: WorldState) -> WorldState:
    import copy

    copied = copy.deepcopy(state)
    copied.quarter = 0
    return copied


def _baseline_outcome(
    state: WorldState, firms: FirmAttributes, segments: SegmentAttributes, graphs: Graphs
) -> QuarterOutcome:
    """Common pre-policy Regime-F outcome used by reward and backfire comparisons."""
    alive = state.alive.astype(np.float64)
    n_alive = max(float(alive.sum()), 1.0)
    revenue = state.revenue * alive
    compliant = state.y * alive
    by_tercile = tuple(
        float(
            np.sum(compliant * (firms.size_tercile == tercile))
            / max(np.sum(alive * (firms.size_tercile == tercile)), 1.0)
        )
        for tercile in (0, 1, 2)
    )
    utility = (
        CONSTANTS.quality_weight * firms.quality[None, :]
        + THETA_STAR.mu_privacy * segments.privacy[:, None] * compliant[None, :]
    )
    masked = np.where(graphs.market_mask & state.alive[None, :], utility, -np.inf)
    inclusive_value = logsumexp(masked, axis=1)
    inclusive_value = np.where(np.isfinite(inclusive_value), inclusive_value, 0.0)
    return QuarterOutcome(
        compliance_rate=float(compliant.sum() / n_alive),
        compliance_rate_weighted=float(
            np.sum(compliant * revenue) / max(float(revenue.sum()), 1e-9)
        ),
        compliance_by_tercile=(by_tercile[0], by_tercile[1], by_tercile[2]),
        hhi=hhi(state.revenue, state.alive),
        mean_trust=float(np.sum(segments.weight * state.trust) / segments.weight.sum()),
        consumer_surplus=float(np.sum(segments.weight * inclusive_value)),
        exit_rate_cum=float(1.0 - alive.sum() / firms.n),
        enforcement_cost=0.0,
        n_audits=0,
    )


def _do_interventions(
    cfg: RegWorldConfig,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs_true: Graphs,
    rollout: np.ndarray,
    observed_traj: Trajectory,
    t_start_obs: np.ndarray,
) -> tuple[Path, Path]:
    """§7.10: true total-regulation-onset ATT (paired counterfactual runs) and the
    analytic per-firm audit CATE along the observed trajectory."""
    from regworld.causal.did import group_time_att

    n_do = max(2, min(cfg.causal.n_do_seeds, 16 if cfg.profile_name == "smoke" else 64))
    horizon = cfg.horizon_quarters
    obs_q = cfg.observed_quarters
    per_quarter = np.zeros((n_do, obs_q))
    att_cells = np.zeros(n_do)
    terminal = np.zeros(n_do)
    did_truth = np.zeros(n_do)
    for k in range(n_do):
        seed = cfg.seed + 60_000 + k
        traj_t, ts = dgp_history.run_history(
            cfg,
            firms,
            segments,
            graphs_true,
            seed,
            rollout=rollout,
        )
        traj_c, _ = dgp_history.run_history(
            cfg,
            firms,
            segments,
            graphs_true,
            seed,
            rollout=rollout,
            force_never_treated=True,
        )
        y_t = np.stack([c["compliant"] for c in traj_t.covariates])  # (T, F)
        y_c = np.stack([c["compliant"] for c in traj_c.covariates])
        diff = y_t - y_c
        per_quarter[k] = diff[:obs_q].mean(axis=1)
        # ATT over observed post-treatment cells, matching the DiD estimand
        cells = []
        for t in range(obs_q):
            post = ts <= t
            if post.any():
                cells.append(diff[t, post].mean())
        att_cells[k] = float(np.mean(cells)) if cells else 0.0
        terminal[k] = diff[horizon - 1].mean()
        # The DiD-commensurable estimand: the same clean-comparison group-time
        # estimator Stage 5 runs, applied to the TRUE panel of this rollout run.
        # Under cross-region interference (peer + macro spillovers reach the
        # not-yet-treated controls) this deliberately differs from the
        # nobody-ever-treated ATT above; Stage 5 grades the DiD against THIS.
        cohort_1b = np.where(
            ts >= dgp_history.NEVER_TREATED, -1, ts + 1
        )  # panel convention: treatment_quarter = t_start + 1, -1 = never
        rows_y, rows_q, rows_g = [], [], []
        for q in range(1, obs_q):
            covs_q = traj_t.covariates[q - 1]
            alive_q = covs_q["alive"] > 0.5
            rows_y.append(y_t[q - 1][alive_q])
            rows_q.append(np.full(int(alive_q.sum()), q, dtype=np.int64))
            rows_g.append(cohort_1b[alive_q])
        did_truth[k], _, _ = group_time_att(
            np.concatenate(rows_y), np.concatenate(rows_q), np.concatenate(rows_g)
        )
    # analytic audit CATE along the observed trajectory (quarters with enforcement on)
    boost = CONSTANTS.own_audit_boost
    cates = np.zeros(firms.n)
    counts = np.zeros(firms.n)
    log_size = np.log(np.maximum(firms.size, 1e-9))
    for t in range(1, obs_q):
        covs = observed_traj.covariates[t]
        audited_prev = observed_traj.covariates[t - 1]["audited"]
        q_obs = covs["perceived_risk"]
        q_no = q_obs / (1.0 + boost * audited_prev)
        active = (t_start_obs <= t) & (covs["alive"] > 0.5)
        common = dict(
            theta=THETA_STAR,
            const=CONSTANTS,
            kappa=covs["cost_share"],
            n_peer=covs["neighbor_compliant_share"],
            m_assoc=covs["assoc_compliant_share"],
            log_size=log_size,
            x_privacy=covs["privacy_rev_share"],
            phi=covs["phase_phi"],
            y_prev=covs["compliant_lag"],
            z=firms.z,
            sticky=True,
            interacted=cfg.dgp.decision_rule == "logit_interacted",
        )
        p1 = expit(firm_utility(q=q_no * (1.0 + boost), **common))
        p0 = expit(firm_utility(q=q_no, **common))
        cates += np.where(active, p1 - p0, 0.0)
        counts += active
    cate = np.divide(cates, counts, out=np.zeros_like(cates), where=counts > 0)
    df = pl.DataFrame(
        {
            "firm_id": np.arange(firms.n, dtype=np.int64),
            "cate_audit": cate,
            "size_tercile": firms.size_tercile,
            "ever_active": (counts > 0),
        }
    )
    p1_path = store.write_oracle_parquet(cfg, "do_interventions", df)
    summary = {
        "tau_true_onset_att": float(att_cells.mean()),
        "tau_true_onset_att_se": float(att_cells.std(ddof=1) / np.sqrt(n_do)),
        "tau_true_onset_per_quarter": per_quarter.mean(axis=0).tolist(),
        "tau_true_terminal": float(terminal.mean()),
        "tau_true_audit_ate": float(cate[counts > 0].mean()) if (counts > 0).any() else 0.0,
        "tau_true_audit_cate_by_tercile": [
            float(cate[(firms.size_tercile == k) & (counts > 0)].mean())
            if ((firms.size_tercile == k) & (counts > 0)).any()
            else 0.0
            for k in (0, 1, 2)
        ],
        "tau_did_truth": float(did_truth.mean()),
        "tau_did_truth_se": float(did_truth.std(ddof=1) / np.sqrt(n_do)),
        "tau_interference_gap": float(att_cells.mean() - did_truth.mean()),
        "did_truth_estimand": (
            "group_time_att (not-yet-treated controls) on the true panel of the "
            "as-scheduled rollout; differs from tau_true_onset_att because peer and "
            "macro-trust spillovers reach the controls (cross-region interference)"
        ),
        "n_do_seeds": n_do,
        "estimand": "total_regulation_onset_att",
        "regulation_off_enforcement": 0.0,
        "regulation_on_enforcement": dgp_history.REGIME_P_LEVERS.enforcement,
        "onset_also_activates_phase_and_compliance_cost": True,
    }
    p2_path = store.write_oracle_json(cfg, "true_effects", summary)
    return p1_path, p2_path


def generate_ground_truth(cfg: RegWorldConfig) -> GenerationResult:
    """The one entry point (§9): writes observed/ (everything may read) and
    oracle/ (evaluation only)."""
    rng = np.random.default_rng(cfg.seed)
    firms = generate_firms(cfg, rng)
    segments, seg_pref = generate_segments(cfg, firms, rng)
    reg = build_graphs(
        cfg,
        rng,
        size=firms.size,
        sector=firms.sector,
        z=firms.z,
        association=firms.association,
        seg_pref=seg_pref,
    )
    g_true = reg.runtime(
        observed=False, n_firms=firms.n, n_segments=cfg.population.n_consumer_segments
    )
    log.info(
        "world generated: %d firms, %d segments; running Regime P",
        firms.n,
        cfg.population.n_consumer_segments,
    )
    rollout_rng = np.random.default_rng(cfg.seed + 90_001)
    rollout = dgp_history.draw_rollout(cfg, rollout_rng)
    traj, t_start = dgp_history.run_history(cfg, firms, segments, g_true, cfg.seed, rollout=rollout)

    obs_rng = np.random.default_rng(cfg.seed + 1)
    observed_paths = [
        store.write_observed(cfg, "firm_registry", obs.firm_registry(firms, obs_rng)),
        store.write_observed(cfg, "firm_panel", obs.firm_panel(cfg, traj, firms, t_start, obs_rng)),
        store.write_observed(cfg, "aggregate_series", obs.aggregate_series(cfg, traj, obs_rng)),
        store.write_observed(
            cfg, "consumer_survey", obs.consumer_survey(cfg, traj, segments, obs_rng)
        ),
        store.write_observed(cfg, "market", obs.market_stats(cfg, traj, firms)),
    ]
    graph_paths = _write_graph_edges(cfg, reg)
    observed_paths += [p for p in graph_paths if "observed" in str(p)]

    sealed_paths = [p for p in graph_paths if "oracle" in str(p)]
    sealed_paths.append(store.write_oracle_json(cfg, "theta_star", theta_star_dict()))
    sealed_paths.append(
        store.write_oracle_parquet(
            cfg, "regime_p_full", obs.regime_p_full(cfg, traj, firms, t_start)
        )
    )
    # graph metrics incl. assortativity-by-z: the homophily sanity check (§7.2)
    metrics_true = analyze_graphs(reg, observed=False, z=firms.z)
    metrics_obs = analyze_graphs(reg, observed=True, z=None)
    sealed_paths.append(store.write_oracle_json(cfg, "graph_metrics_true", metrics_true))
    graphs_dir = Path(cfg.paths.graphs)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    import json

    (graphs_dir / "metrics_observed.json").write_text(json.dumps(metrics_obs, indent=2))

    log.info(
        "Regime P done (terminal compliance %.3f); writing Regime F truth grid",
        traj.outcomes[-1].compliance_rate,
    )
    # Regime F is a new regulation on the same generated population, not a
    # continuation of Regime P's compliance indicator. Start a fresh episode so
    # CDPA phase-in and compliance both begin at their own quarter zero.
    future_state = initial_state(
        firms,
        segments,
        g_true,
        CONSTANTS,
        np.random.default_rng(cfg.seed + 30_000),
    )
    future_baseline = _baseline_outcome(future_state, firms, segments, g_true)
    sealed_paths.append(
        _regime_f_truth(cfg, firms, segments, g_true, future_state, future_baseline)
    )
    p1, p2 = _do_interventions(cfg, firms, segments, g_true, rollout, traj, t_start)
    sealed_paths += [p1, p2]
    # The planted confounder and true continuous size/cost, sealed for Stage-5
    # grading only (conditioning on z is the "full" control set that closes the
    # backdoor the observed decile leaves open). Never joined into observed data.
    sealed_paths.append(
        store.write_oracle_parquet(
            cfg,
            "firm_confounders",
            pl.DataFrame(
                {
                    "firm_id": np.arange(firms.n, dtype=np.int64),
                    "capacity_z": firms.z.astype(np.float64),
                    "size": firms.size.astype(np.float64),
                    "cost_coef": firms.cost_coef.astype(np.float64),
                    "size_tercile": firms.size_tercile.astype(np.int64),
                }
            ),
        )
    )
    log.info(
        "generation complete: %d observed, %d oracle artifacts",
        len(observed_paths),
        len(sealed_paths),
    )
    return GenerationResult(observed_paths=observed_paths, sealed_paths=sealed_paths)
