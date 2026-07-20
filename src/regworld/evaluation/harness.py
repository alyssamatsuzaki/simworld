"""Shared plumbing for the §11 metric families: context loading and rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from regworld.models.world_model import WorldModel
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.training.datamodule import EmulatorSequences
from regworld.training.losses import symexp
from regworld.types import RegWorldConfig


@dataclass
class EvalContext:
    cfg: RegWorldConfig
    model: WorldModel
    meta: dict[str, Any]
    heldout: EmulatorSequences
    batch: dict[str, torch.Tensor]  # every held-out episode, full length
    episodes: list[int]


def load_context(cfg: RegWorldConfig, arch: str | None = None) -> EvalContext:
    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, arch or cfg.emulator.arch))
    heldout = EmulatorSequences(cfg, "heldout")
    names = ("firm", "segment", "aggregate", "action", "reward", "cont")
    stacked = {
        name: torch.as_tensor(
            np.stack([heldout.episode_arrays(e)[name] for e in heldout.episodes]),
            dtype=torch.float32,
        )
        for name in names
    }
    return EvalContext(
        cfg=cfg,
        model=model,
        meta=meta,
        heldout=heldout,
        batch=stacked,
        episodes=list(heldout.episodes),
    )


@torch.no_grad()
def open_loop_natural(
    model: WorldModel,
    batch: dict[str, torch.Tensor],
    *,
    burn_in: int,
    horizon: int,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Open-loop rollout decoded to natural units.

    Returns (aggregates (B, K, A), node probabilities (B, K, N), start index).
    """
    agg_symlog, node_logits, start = model.open_loop(
        batch, burn_in=burn_in, horizon=horizon, generator=generator
    )
    return (
        symexp(agg_symlog).numpy(),
        torch.sigmoid(node_logits).numpy(),
        start,
    )


@torch.no_grad()
def sampled_open_loop(
    model: WorldModel,
    batch: dict[str, torch.Tensor],
    *,
    burn_in: int,
    horizon: int,
    n_samples: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Stochastic ensemble of open-loop rollouts: (S, B, K, A) natural units."""
    samples = []
    for s in range(n_samples):
        generator = torch.Generator().manual_seed(base_seed + s)
        agg, _, _ = open_loop_natural(
            model, batch, burn_in=burn_in, horizon=horizon, generator=generator
        )
        samples.append(agg)
    return np.stack(samples)


def episode_slice(batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {name: values[index : index + 1] for name, values in batch.items()}
