"""§10 Stage 11 — the scenario cube: (policy x posterior-draw x seed) rollouts.

The trained GraphRSSM emulator already marginalizes theta through training-time
domain randomization (§10 Stages 6+7: episodes are collected across the Stage-4
posterior). Its remaining parametric uncertainty at inference time therefore
shows up through the categorical-latent sampling rather than by re-drawing
theta, so a cube cell is one distinctly-seeded imagined rollout: for each
policy, ``posterior_draws x n_seeds`` of them, each driven by its own
``torch.Generator`` seed (via ``EmulatorEnv.reset(seed=...)``) so any single
cell is reproducible independent of batching or scheduling order.

Ray usage mirrors ``training.datamodule.build_dataset``: actors that each hold
one loaded emulator process a batch of cells in sequence, wrapped in a bounded
``ray.get(..., timeout=...)``, with a serial fallback used whenever Ray is
unavailable, below the size threshold, or its worker bootstrap fails (broken
in this environment: ``raylet: ModuleNotFoundError: No module named 'ray'``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from regworld.abm.policies import STATIC_POLICIES
from regworld.environments.emulator_env import EmulatorEnv
from regworld.models.world_model import WorldModel
from regworld.rules import backfire
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.training.datamodule import aggregate_to_outcome
from regworld.types import RegWorldConfig

try:  # Stage 10 (RL) runs in parallel with this stage; guard its registry.
    from regworld.agents.registry import load_policy
except Exception:  # pragma: no cover - Stage 10 artifact/module may not exist yet
    load_policy = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

ActionFn = Callable[[NDArray[np.float32]], NDArray[np.float32]]
CellSpec = tuple[str, int, int]  # (policy name, posterior-draw index, seed index)

# Below this many cells, Ray's scheduling overhead dominates the work itself
# (mirrors the 256-episode threshold in training.datamodule.build_dataset).
RAY_CELL_THRESHOLD = 256


def policy_id(name: str) -> str:
    """Stable hash of the policy name — the cube's ``policy`` coordinate (§8)."""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def cube_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "ensemble"


def _cell_seed(base_seed: int, policy: str, draw_idx: int, seed_idx: int) -> int:
    """Deterministic, distinct seed for one (policy, draw, seed) cell.

    Hashing (not sampling) keeps this reproducible independent of iteration
    order, batch composition, or how many workers process the cube.
    """
    key = f"{base_seed}:{policy}:{draw_idx}:{seed_idx}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _static_action_fn(name: str) -> ActionFn:
    action = STATIC_POLICIES[name].as_array().astype(np.float32)

    def fn(_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return action

    return fn


def resolve_policies(
    cfg: RegWorldConfig, names: Sequence[str]
) -> tuple[dict[str, ActionFn], dict[str, str]]:
    """Resolve each requested policy name to an obs -> action callable.

    Static policies (§10 Stage 10a) resolve locally from
    ``regworld.abm.policies.STATIC_POLICIES``. Learned policies (``rl_ppo``,
    ``rl_dreamer``) come from the Stage-10 registry; if that module is not
    importable, or it has no trained artifact for this name, the policy is
    skipped and noted rather than crashing the ensemble.
    """
    resolved: dict[str, ActionFn] = {}
    skipped: dict[str, str] = {}
    for name in names:
        if name in STATIC_POLICIES:
            resolved[name] = _static_action_fn(name)
            continue
        if load_policy is None:
            skipped[name] = "unavailable (Stage 10 artifact not present)"
            continue
        fn: ActionFn | None
        try:
            fn = load_policy(cfg, name)
        except Exception as exc:  # pragma: no cover - artifact-dependent
            log.warning("policy %r failed to load: %s", name, exc)
            fn = None
        if fn is None:
            skipped[name] = "unavailable (Stage 10 artifact not present)"
        else:
            resolved[name] = fn
    return resolved, skipped


def _build_cells(policy_names: Sequence[str], n_draws: int, n_seeds: int) -> list[CellSpec]:
    return [
        (policy, draw_idx, seed_idx)
        for policy in policy_names
        for draw_idx in range(n_draws)
        for seed_idx in range(n_seeds)
    ]


def _rollout_cell(
    cfg: RegWorldConfig,
    model: WorldModel,
    meta: dict[str, Any],
    action_fn: ActionFn,
    seed: int,
) -> dict[str, Any]:
    """Roll one scenario-cube cell to horizon under a policy; return its terminal row."""
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    env = EmulatorEnv(cfg, model=model, meta=meta)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    quarters = 0
    while not (terminated or truncated):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        quarters += 1

    n_firms = int(meta["extras"]["n_firms"])
    # EmulatorEnv's Gym API only exposes a policy-learning observation, not the
    # raw natural-unit aggregate row; its ``_aggregates`` instance attribute
    # (set every step()) is the terminal state we actually need here.
    terminal = aggregate_to_outcome(np.asarray(env._aggregates, dtype=np.float64), n_firms)
    baseline = aggregate_to_outcome(
        np.asarray(meta["initial"]["aggregate"], dtype=np.float64), n_firms
    )
    return {
        "compliance_rate": terminal.compliance_rate,
        "compliance_rate_weighted": terminal.compliance_rate_weighted,
        "hhi": terminal.hhi,
        "mean_trust": terminal.mean_trust,
        "consumer_surplus": terminal.consumer_surplus,
        "exit_rate": terminal.exit_rate_cum,
        "enforcement_cost": terminal.enforcement_cost,
        "reward": total_reward,
        "backfire": bool(backfire(terminal, baseline)),
        "collapsed": bool(terminated),
        "quarters": quarters,
    }


def _run_cells_serial(
    cfg: RegWorldConfig,
    model: WorldModel,
    meta: dict[str, Any],
    actions: dict[str, ActionFn],
    cells: Sequence[CellSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, draw_idx, seed_idx in cells:
        seed = _cell_seed(cfg.seed, policy, draw_idx, seed_idx)
        row = _rollout_cell(cfg, model, meta, actions[policy], seed)
        row.update(
            {
                "policy": policy,
                "policy_id": policy_id(policy),
                "draw": draw_idx,
                "seed_idx": seed_idx,
                "seed": seed,
            }
        )
        rows.append(row)
    return rows


def _run_cells_ray(
    cfg: RegWorldConfig,
    cell_batches: list[list[CellSpec]],
    policy_names: Sequence[str],
) -> list[dict[str, Any]] | None:
    """Ray Core actor path: one loaded emulator per actor, batches run in sequence.

    Returns ``None`` (never raises) on any failure so the caller falls back to
    the serial path — Ray's worker bootstrap is environment-sensitive and a
    broken one hangs rather than raising, so a bounded ``ray.get(timeout=...)``
    plus a broad except is what keeps this from stalling the run.
    """
    try:
        import ray

        @ray.remote
        class _EmulatorActor:
            def __init__(self, cfg: RegWorldConfig, policy_names: Sequence[str]) -> None:
                self._cfg = cfg
                self._model, self._meta = load_checkpoint(
                    checkpoint_path(cfg.paths.root, cfg.emulator.arch)
                )
                self._actions, _ = resolve_policies(cfg, policy_names)

            def run_batch(self, batch: list[CellSpec]) -> list[dict[str, Any]]:
                return _run_cells_serial(self._cfg, self._model, self._meta, self._actions, batch)

        ray.init(
            num_cpus=cfg.compute.num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            configure_logging=False,
        )
        n_actors = max(1, min(len(cell_batches), cfg.compute.num_cpus or len(cell_batches)))
        actors = [_EmulatorActor.remote(cfg, policy_names) for _ in range(n_actors)]  # type: ignore[attr-defined]
        futures = [
            actors[i % n_actors].run_batch.remote(batch) for i, batch in enumerate(cell_batches)
        ]
        results: list[dict[str, Any]] = []
        for chunk in ray.get(futures, timeout=1800.0):
            results.extend(chunk)
        ray.shutdown()
        return results
    except Exception:  # pragma: no cover - environment-dependent fallback
        log.warning("Ray ensemble collection failed; falling back to serial", exc_info=True)
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
        return None


def build_cube(
    cfg: RegWorldConfig, model: WorldModel, meta: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Build the ``(policy x posterior-draw x seed)`` terminal-outcome scenario cube."""
    from regworld.pipeline import Degraded

    actions, skipped = resolve_policies(cfg, cfg.ensemble.policies)
    if not actions:
        raise Degraded("no policy in cfg.ensemble.policies resolved to a rollout", outputs=[])

    cells = _build_cells(list(actions), cfg.ensemble.posterior_draws, cfg.ensemble.n_seeds)
    batch_size = max(cfg.ensemble.batch_size, 1)
    batches = [cells[i : i + batch_size] for i in range(0, len(cells), batch_size)]

    use_ray = cfg.compute.name.startswith("ray") and len(cells) >= RAY_CELL_THRESHOLD
    rows: list[dict[str, Any]] | None = None
    if use_ray:
        rows = _run_cells_ray(cfg, batches, list(actions))
    if rows is None:
        rows = []
        for batch in batches:
            rows.extend(_run_cells_serial(cfg, model, meta, actions, batch))

    frame = pl.DataFrame(rows)
    return frame, skipped


def _series_mean(series: pl.Series) -> float:
    if series.len() == 0:
        return float("nan")
    return float(np.asarray(series.to_numpy(), dtype=np.float64).mean())


@dataclass
class EnsembleResult:
    cube: Path
    summary: Path
    metrics: dict[str, float]


def run_ensemble(cfg: RegWorldConfig) -> EnsembleResult:
    """§10 Stage 11: build the scenario cube and cross-validate it against the ABM."""
    from regworld.ensemble.validation import run_validation
    from regworld.pipeline import Degraded

    out = cube_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    try:
        model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    except FileNotFoundError as exc:
        raise Degraded(f"no trained emulator checkpoint: {exc}", outputs=[]) from exc

    frame, skipped = build_cube(cfg, model, meta)
    cube_path = out / "cube.parquet"
    frame.write_parquet(cube_path)

    validation = run_validation(cfg, frame, model, meta)

    metrics: dict[str, float] = {
        "n_cells": float(frame.height),
        "n_policies_included": float(frame["policy"].n_unique()) if frame.height else 0.0,
        "n_policies_skipped": float(len(skipped)),
        "mean_reward": _series_mean(frame["reward"]) if frame.height else float("nan"),
        "backfire_rate": _series_mean(frame["backfire"]) if frame.height else float("nan"),
        "coverage": validation.coverage,
    }
    summary = {
        "profile": cfg.profile_name,
        "policies_included": sorted(frame["policy"].unique().to_list()) if frame.height else [],
        "policies_skipped": skipped,
        "metrics": metrics,
        "validation": {
            "coverage": validation.coverage,
            "n_validated": validation.n_validated,
            "per_policy": validation.per_policy,
        },
        "cube_path": str(cube_path),
        "validation_path": str(validation.path),
    }
    summary_path = out / "ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return EnsembleResult(cube=cube_path, summary=summary_path, metrics=metrics)


__all__ = [
    "RAY_CELL_THRESHOLD",
    "ActionFn",
    "CellSpec",
    "EnsembleResult",
    "build_cube",
    "cube_dir",
    "policy_id",
    "resolve_policies",
    "run_ensemble",
]
