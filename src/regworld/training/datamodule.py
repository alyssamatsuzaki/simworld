"""Emulator training data: domain-randomized tensorized-ABM rollouts in Zarr.

Domain randomization is not optional (§10 Stages 6+7): theta is drawn from the
Stage-4 posterior and policies from a mixture of random-uniform, scripted,
sinusoidal-sweep, and piecewise-constant schedules — without it the emulator
memorizes one policy and every scenario-grid number is fiction. (A second round
with RL-policy data is added in Phase 6.)

Layout: one Zarr group with ``(episode, quarter, node, feature)`` arrays plus an
``episodes.parquet`` manifest (seed, theta draw, policy kind, split). Row 0 of
every episode is the deterministic initial world state; ``action[t]`` is the
lever vector that produced row ``t``.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from regworld import rules
from regworld.abm.model import ObservedWorld, load_observed_world
from regworld.abm.policies import STATIC_POLICIES
from regworld.abm.tensorized import rollout_tensorized
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

ACTION_LOW = np.array([0.0, -1.0, 0.0, 0.0])
ACTION_HIGH = np.ones(4)
AGGREGATE_BASE = (
    "compliance_rate",
    "compliance_rate_weighted",
    "hhi",
    "mean_trust",
    "consumer_surplus",
    "exit_rate_cum",
    "audit_rate",
    "penalty_rate",
)
POLICY_KINDS = ("static_random", "scripted", "sinusoid", "piecewise")
SPLITS = ("train", "val", "heldout")


def aggregate_names(cfg: RegWorldConfig) -> list[str]:
    return [
        *AGGREGATE_BASE,
        *[f"sector_{k}" for k in range(cfg.population.n_sectors)],
        *[f"decile_{d}" for d in range(10)],
    ]


def aggregate_dim(cfg: RegWorldConfig) -> int:
    return len(AGGREGATE_BASE) + cfg.population.n_sectors + 10


def dataset_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "emulator" / "dataset"


def load_graph_bundle(cfg: RegWorldConfig) -> tuple[dict[str, torch.Tensor], Any]:
    """Static node features + GraphTemplate from Stage 2's ``hetero_observed.pt``."""
    from regworld.models.gnn import GraphTemplate

    path = Path(cfg.paths.graphs) / "hetero_observed.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `make graphs` first")
    data = torch.load(path, weights_only=False)
    static = {ntype: data[ntype].x.float() for ntype in data.node_types}
    return static, GraphTemplate.from_hetero_data(data)


def load_theta_draws(cfg: RegWorldConfig) -> np.ndarray:
    """Posterior draw matrix (n_draws, |Theta|) from Stage 4, in field order.

    ``beta_capacity`` stays at 0.0: the latent capacity confounder is never
    fitted, so the emulator never sees a world its calibration could not see.
    """
    import arviz as az

    path = Path(cfg.paths.root) / "calibration" / "posterior.nc"
    if not path.is_file():
        raise FileNotFoundError(f"calibrated posterior not found: {path}; run `make calibrate`")
    idata = az.from_netcdf(path)
    names = list(rules.Theta.__dataclass_fields__)
    defaults = np.array([getattr(rules.Theta(), n) for n in names])
    columns: list[np.ndarray | None] = []
    for name in names:
        if name in idata.posterior.data_vars and name != "beta_capacity":
            columns.append(np.asarray(idata.posterior[name]).reshape(-1))
        else:
            columns.append(None)
    n_draws = max(c.shape[0] for c in columns if c is not None)
    matrix = np.tile(defaults, (n_draws, 1))
    for j, col in enumerate(columns):
        if col is not None:
            matrix[:, j] = col[:n_draws]
    return matrix


def sample_policy_schedule(rng: np.random.Generator, quarters: int) -> tuple[np.ndarray, str]:
    """One (quarters, 4) lever schedule from the domain-randomization mixture."""
    kind = str(rng.choice(POLICY_KINDS, p=[0.3, 0.2, 0.25, 0.25]))
    if kind == "static_random":
        levers = rng.uniform(ACTION_LOW, ACTION_HIGH)
        schedule = np.tile(levers, (quarters, 1))
    elif kind == "scripted":
        name = str(rng.choice(sorted(STATIC_POLICIES)))
        schedule = np.tile(STATIC_POLICIES[name].as_array(), (quarters, 1))
    elif kind == "sinusoid":
        t = np.arange(quarters)[:, None]
        mid = rng.uniform(ACTION_LOW, ACTION_HIGH)
        amp = rng.uniform(0.0, (ACTION_HIGH - ACTION_LOW) / 2.0)
        period = rng.choice([8.0, 12.0, 24.0], size=4)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
        schedule = mid + amp * np.sin(2.0 * np.pi * t / period + phase)
    else:  # piecewise-constant
        n_segments = int(rng.integers(2, 5))
        cuts = np.sort(rng.choice(np.arange(1, quarters), size=n_segments - 1, replace=False))
        bounds = [0, *cuts.tolist(), quarters]
        schedule = np.empty((quarters, 4))
        for lo, hi in itertools.pairwise(bounds):
            schedule[lo:hi] = rng.uniform(ACTION_LOW, ACTION_HIGH)
    return np.clip(schedule, ACTION_LOW, ACTION_HIGH), kind


def _size_deciles(size: np.ndarray) -> np.ndarray:
    if size.size <= 1:
        return np.zeros(size.size, dtype=np.int64)
    return np.digitize(size, np.quantile(size, np.linspace(0.1, 0.9, 9))).astype(np.int64)


def _group_compliance(
    y: np.ndarray, alive: np.ndarray, groups: np.ndarray, n_groups: int
) -> np.ndarray:
    out = np.zeros(n_groups)
    for g in range(n_groups):
        mask = (alive > 0.5) & (groups == g)
        if mask.any():
            out[g] = float(np.mean(y[mask]))
    return out


def initial_frame(cfg: RegWorldConfig, world: ObservedWorld) -> dict[str, np.ndarray]:
    """Row 0: the deterministic pre-policy world state, §7.6 aggregates included."""
    firms, segments, state = world.firms, world.segments, world.initial_state
    const = rules.Constants()
    utility = const.quality_weight * firms.quality[None, :]
    alive = np.ones(firms.n, dtype=bool)
    _, _, consumer_surplus = rules.allocate_spend(
        utility, alive, world.graphs.market_mask, segments
    )
    aggregates = np.zeros(aggregate_dim(cfg))
    aggregates[2] = rules.hhi(state.revenue, alive)
    aggregates[3] = float(np.sum(segments.weight * state.trust) / np.sum(segments.weight))
    aggregates[4] = float(consumer_surplus)
    firm_dynamic = np.zeros((firms.n, 4))
    firm_dynamic[:, 1] = 1.0
    firm_dynamic[:, 2] = firms.base_margin
    return {
        "firm": firm_dynamic,
        "segment": state.trust[:, None].copy(),
        "aggregate": aggregates,
    }


def collect_episode(
    cfg: RegWorldConfig,
    world: ObservedWorld,
    theta: rules.Theta,
    schedule: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """One tensorized rollout -> (T+1)-row arrays matching the Zarr layout."""
    quarters = cfg.horizon_quarters
    firms = world.firms
    const = rules.Constants()
    trajectory = rollout_tensorized(
        cfg,
        world,
        theta,
        rules.PolicyLevers(),
        seed=seed,
        quarters=quarters,
        lever_schedule=schedule,
    )
    frame0 = initial_frame(cfg, world)
    n_rows = quarters + 1
    firm_arr = np.zeros((n_rows, firms.n, 4), dtype=np.float32)
    segment_arr = np.zeros((n_rows, world.segments.weight.size, 1), dtype=np.float32)
    agg_arr = np.zeros((n_rows, aggregate_dim(cfg)), dtype=np.float32)
    action_arr = np.zeros((n_rows, 4), dtype=np.float32)
    reward_arr = np.zeros(n_rows, dtype=np.float32)
    cont_arr = np.ones(n_rows, dtype=np.float32)
    firm_arr[0] = frame0["firm"]
    segment_arr[0] = frame0["segment"]
    agg_arr[0] = frame0["aggregate"]
    action_arr[1:] = schedule.astype(np.float32)

    deciles = _size_deciles(firms.size)
    weights: tuple[float, float, float, float, float, float] = tuple(  # type: ignore[assignment]
        float(getattr(cfg.objective, name)) for name in ("w_c", "w_h", "w_s", "w_e", "w_t", "w_x")
    )
    baseline = aggregate_to_outcome(agg_arr[0], firms.n)
    fine_fraction = min(const.fine_rate, const.fine_cap)
    prev_revenue = world.initial_state.revenue.copy()
    cumulative_audits = 0.0
    max_audits = max(cfg.horizon_quarters * const.audit_budget * firms.n, 1.0)
    for t in range(1, n_rows):
        cov = {k: v.detach().cpu().numpy() for k, v in trajectory.covariates[t - 1].items()}
        outcome = trajectory.outcomes[t - 1]
        y, alive = cov["compliant"], cov["alive"]
        revenue, fined = cov["revenue"], cov["fined"]
        fines = fine_fraction * np.clip(prev_revenue, 0.0, None) * fined
        cost_share = cov["cost_share"]
        margin = firms.base_margin - y * cost_share - fines / np.clip(revenue, 1e-9, None)
        firm_arr[t] = np.stack([y, alive, margin, cost_share], axis=1)
        segment_arr[t, :, 0] = cov["segment_trust"]
        alive_count = max(float(alive.sum()), 1.0)
        n_audits = float(outcome.n_audits.item())
        revenue_total = max(float(np.sum(revenue * alive)), 1e-9)
        agg = np.concatenate(
            [
                [
                    float(outcome.compliance_rate.item()),
                    float(outcome.compliance_rate_weighted.item()),
                    float(outcome.hhi.item()),
                    float(outcome.mean_trust.item()),
                    float(outcome.consumer_surplus.item()),
                    float(outcome.exit_rate_cum.item()),
                    n_audits / alive_count,
                    min(float(fines.sum()) / revenue_total, 1.0),
                ],
                _group_compliance(y, alive, firms.sector, cfg.population.n_sectors),
                _group_compliance(y, alive, deciles, 10),
            ]
        )
        agg_arr[t] = agg
        reward_arr[t] = rules.regulator_reward(
            aggregate_to_outcome(agg, firms.n), baseline, weights, const, firms.n
        )
        cumulative_audits += n_audits
        budget_remaining = 1.0 - cumulative_audits / max_audits
        collapsed = agg[5] > 0.40 or (t > 12 and agg[0] < 0.05 and budget_remaining <= 0.0)
        if collapsed:
            cont_arr[t] = 0.0
        prev_revenue = revenue
    return {
        "firm": firm_arr,
        "segment": segment_arr,
        "aggregate": agg_arr,
        "action": action_arr,
        "reward": reward_arr,
        "cont": cont_arr,
    }


def aggregate_to_outcome(agg: np.ndarray, n_firms: int) -> rules.QuarterOutcome:
    """Rebuild the reward-relevant QuarterOutcome fields from an aggregate row."""
    const = rules.Constants()
    alive_count = n_firms * (1.0 - float(agg[5]))
    n_audits = round(float(agg[6]) * max(alive_count, 1.0))
    return rules.QuarterOutcome(
        compliance_rate=float(agg[0]),
        compliance_rate_weighted=float(agg[1]),
        compliance_by_tercile=(0.0, 0.0, 0.0),
        hhi=float(agg[2]),
        mean_trust=float(agg[3]),
        consumer_surplus=float(agg[4]),
        exit_rate_cum=float(agg[5]),
        enforcement_cost=n_audits * const.audit_unit_cost,
        n_audits=n_audits,
    )


def _episode_task(
    cfg: RegWorldConfig,
    world: ObservedWorld,
    theta_rows: np.ndarray,
    episodes: list[int],
    base_seed: int,
) -> list[tuple[int, dict[str, np.ndarray], str, int]]:
    theta_names = list(rules.Theta.__dataclass_fields__)
    results = []
    for episode in episodes:
        rng = np.random.default_rng(base_seed + episode)
        draw = int(rng.integers(theta_rows.shape[0]))
        theta = rules.Theta(**dict(zip(theta_names, theta_rows[draw].tolist(), strict=True)))
        schedule, kind = sample_policy_schedule(rng, cfg.horizon_quarters)
        arrays = collect_episode(cfg, world, theta, schedule, seed=base_seed + episode)
        results.append((episode, arrays, kind, draw))
    return results


def build_dataset(
    cfg: RegWorldConfig,
    *,
    n_episodes: int | None = None,
    use_ray: bool | None = None,
) -> Path:
    """Collect the training corpus and write the Zarr group + manifest."""
    import zarr

    n_episodes = cfg.emulator.train_episodes if n_episodes is None else n_episodes
    out = dataset_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    world = load_observed_world(cfg)
    theta_rows = load_theta_draws(cfg)
    base_seed = cfg.seed + 90_000
    if use_ray is None:
        # Serial below dev scale: a smoke corpus collects in seconds, and Ray's
        # worker bootstrap is environment-sensitive (see DEVIATIONS 2026-07-20).
        use_ray = cfg.compute.name.startswith("ray") and n_episodes >= 256

    batches = [list(range(i, min(i + 8, n_episodes))) for i in range(0, n_episodes, 8)]
    results: list[tuple[int, dict[str, np.ndarray], str, int]] = []
    if use_ray:
        try:
            import ray

            ray.init(
                num_cpus=cfg.compute.num_cpus,
                include_dashboard=False,
                ignore_reinit_error=True,
                configure_logging=False,
            )
            task = ray.remote(_episode_task)
            world_ref = ray.put(world)
            theta_ref = ray.put(theta_rows)
            futures = [task.remote(cfg, world_ref, theta_ref, b, base_seed) for b in batches]
            # Bounded wait: a broken worker bootstrap hangs rather than raising,
            # so convert it into the serial fallback instead of stalling the gate.
            for chunk in ray.get(futures, timeout=1800.0):
                results.extend(chunk)
            ray.shutdown()
        except Exception:  # pragma: no cover - environment-dependent fallback
            log.warning("Ray collection failed; falling back to serial", exc_info=True)
            try:
                import ray

                ray.shutdown()
            except Exception:
                pass
            results = []
    if not results:
        for batch in batches:
            results.extend(_episode_task(cfg, world, theta_rows, batch, base_seed))
    results.sort(key=lambda item: item[0])

    quarters = cfg.horizon_quarters
    n_rows = quarters + 1
    root = zarr.open_group(str(out / "episodes.zarr"), mode="w")
    shapes = {
        "firm": (n_episodes, n_rows, world.firms.n, 4),
        "segment": (n_episodes, n_rows, world.segments.weight.size, 1),
        "aggregate": (n_episodes, n_rows, aggregate_dim(cfg)),
        "action": (n_episodes, n_rows, 4),
        "reward": (n_episodes, n_rows),
        "cont": (n_episodes, n_rows),
    }
    arrays = {
        name: root.zeros(name, shape=shape, chunks=(1, *shape[1:]), dtype="f4")
        for name, shape in shapes.items()
    }
    for episode, episode_arrays, _, _ in results:
        for name, values in episode_arrays.items():
            arrays[name][episode] = values
    frame0 = initial_frame(cfg, world)
    root.array("initial_firm", frame0["firm"].astype(np.float32))
    root.array("initial_segment", frame0["segment"].astype(np.float32))
    root.array("initial_aggregate", frame0["aggregate"].astype(np.float32))
    root.attrs.update(
        {
            "aggregate_names": aggregate_names(cfg),
            "n_firms": world.firms.n,
            "quarters": quarters,
            "action_low": ACTION_LOW.tolist(),
            "action_high": ACTION_HIGH.tolist(),
        }
    )
    manifest = pl.DataFrame(
        {
            "episode": [r[0] for r in results],
            "policy_kind": [r[2] for r in results],
            "theta_draw": [r[3] for r in results],
            "seed": [base_seed + r[0] for r in results],
            "split": [SPLITS[0 if e % 10 < 8 else (1 if e % 10 == 8 else 2)] for e, *_ in results],
        }
    )
    manifest.write_parquet(out / "episodes.parquet")
    (out / "dataset_meta.json").write_text(
        json.dumps({"n_episodes": n_episodes, "profile": cfg.profile_name}, indent=2)
    )
    log.info("emulator dataset: %d episodes -> %s", n_episodes, out)
    return out


class EmulatorSequences:
    """Random (episode, window) sequence batches from the Zarr corpus."""

    def __init__(self, cfg: RegWorldConfig, split: str = "train") -> None:
        import zarr

        out = dataset_dir(cfg)
        self.root = zarr.open_group(str(out / "episodes.zarr"), mode="r")
        manifest = pl.read_parquet(out / "episodes.parquet")
        self.episodes = manifest.filter(pl.col("split") == split)["episode"].to_list()
        if not self.episodes:
            raise ValueError(f"no episodes in split {split!r}")
        self.manifest = manifest
        self.n_rows = int(self.root["aggregate"].shape[1])

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int, seq_len: int
    ) -> dict[str, torch.Tensor]:
        seq_len = min(seq_len, self.n_rows)
        names = ("firm", "segment", "aggregate", "action", "reward", "cont")
        stacks: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for _ in range(batch_size):
            episode = int(rng.choice(self.episodes))
            start = int(rng.integers(0, self.n_rows - seq_len + 1))
            for name in names:
                stacks[name].append(self.root[name][episode, start : start + seq_len])
        return {
            name: torch.as_tensor(np.stack(values), dtype=torch.float32)
            for name, values in stacks.items()
        }

    def episode_arrays(self, episode: int) -> dict[str, np.ndarray]:
        names = ("firm", "segment", "aggregate", "action", "reward", "cont")
        return {name: np.asarray(self.root[name][episode]) for name in names}

    def initial_arrays(self) -> dict[str, np.ndarray]:
        return {
            "firm": np.asarray(self.root["initial_firm"]),
            "segment": np.asarray(self.root["initial_segment"]),
            "aggregate": np.asarray(self.root["initial_aggregate"]),
        }


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "AGGREGATE_BASE",
    "EmulatorSequences",
    "aggregate_dim",
    "aggregate_names",
    "build_dataset",
    "collect_episode",
    "dataset_dir",
    "initial_frame",
    "load_graph_bundle",
    "load_theta_draws",
    "sample_policy_schedule",
]
