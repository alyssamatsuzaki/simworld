"""GraphRSSM training loop (§10 Stages 6+7).

Teacher-forced posterior training plus the open-loop imagination loss (roll the
prior ``imag_horizon`` steps and penalize drift — the number that actually
matters). AdamW at ``lr`` with a cosine schedule, gradient clipping at
``grad_clip``. ``torch.compile`` stays behind the config flag, off by default.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from regworld.models.world_model import WorldModel, build_world_model
from regworld.training import datamodule
from regworld.training.checkpoint import checkpoint_path, save_checkpoint
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    arch: str
    checkpoint: Path
    summary: Path
    metrics: dict[str, float] = field(default_factory=dict)


def _constructor_kwargs(cfg: RegWorldConfig, arch: str, aggregate_dim: int) -> dict[str, object]:
    em = cfg.emulator
    return {
        "arch": arch,
        "aggregate_dim": aggregate_dim,
        "action_dim": 4,
        "deter_dim": em.deter_dim,
        "hidden_dim": em.hidden_dim,
        "latent_categories": em.latent_categories,
        "latent_classes": em.latent_classes,
        "gnn_layers": em.gnn_layers,
        "kl_free": em.kl_free,
        "kl_balance": em.kl_balance,
        "stochastic_level": em.stochastic_level,
    }


def total_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(losses.values())).sum()


def reconstruction_loss(losses: dict[str, torch.Tensor]) -> float:
    """The overfit-one-batch criterion: everything except the free-bits KL floor."""
    return float(sum(v.item() for k, v in losses.items() if k != "kl"))


def train_step(
    model: WorldModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    burn_in: int,
    horizon: int,
    grad_clip: float,
) -> dict[str, torch.Tensor]:
    losses = model.observe_losses(batch)
    losses.update(model.imagination_losses(batch, burn_in=burn_in, horizon=horizon))
    loss = total_loss(losses)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return losses


def train_world_model(
    cfg: RegWorldConfig,
    *,
    arch: str | None = None,
    train_steps: int | None = None,
) -> TrainResult:
    """Train one arch on the shared corpus; returns checkpoint + summary paths."""
    arch = arch or cfg.emulator.arch
    em = cfg.emulator
    steps = train_steps if train_steps is not None else em.train_steps
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + sum(ord(c) for c in arch))

    corpus_dir = datamodule.dataset_dir(cfg)
    if not (corpus_dir / "episodes.zarr").exists():
        log.info("emulator dataset missing; collecting %d episodes", em.train_episodes)
        datamodule.build_dataset(cfg)
    train_data = datamodule.EmulatorSequences(cfg, "train")
    val_data = datamodule.EmulatorSequences(cfg, "val")
    static, template = datamodule.load_graph_bundle(cfg)
    agg_dim = datamodule.aggregate_dim(cfg)
    model = build_world_model(cfg, static, template, aggregate_dim=agg_dim, arch=arch)
    if em.compile:  # pragma: no cover - off by default, recurrent code needs coaxing
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(model.parameters(), lr=em.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    started = time.time()
    history: list[dict[str, float]] = []
    for step in range(steps):
        batch = train_data.sample_batch(rng, em.batch_size, em.seq_len + 1)
        losses = train_step(
            model,
            batch,
            optimizer,
            burn_in=em.burn_in,
            horizon=em.imag_horizon,
            grad_clip=em.grad_clip,
        )
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            snapshot = {k: float(v.item()) for k, v in losses.items()}
            snapshot["step"] = float(step)
            history.append(snapshot)
            log.info(
                "%s step %d/%d recon %.4f kl %.4f imag %.4f",
                arch,
                step,
                steps,
                reconstruction_loss(losses),
                snapshot.get("kl", 0.0),
                snapshot.get("imag_aggregate", 0.0),
            )

    model.eval()
    with torch.no_grad():
        val_batch = val_data.sample_batch(rng, em.batch_size, em.seq_len + 1)
        val_losses = model.observe_losses(val_batch)
        val_losses.update(
            model.imagination_losses(val_batch, burn_in=em.burn_in, horizon=em.imag_horizon)
        )
    metrics = {f"val_{k}": float(v.item()) for k, v in val_losses.items()}
    metrics["val_total"] = float(total_loss(val_losses).item())
    metrics["train_seconds"] = time.time() - started
    metrics["parameters"] = float(sum(p.numel() for p in model.parameters()))

    initial = {
        k: torch.as_tensor(v, dtype=torch.float32) for k, v in train_data.initial_arrays().items()
    }
    out = save_checkpoint(
        checkpoint_path(cfg.paths.root, arch),
        model,
        constructor=_constructor_kwargs(cfg, arch, agg_dim),
        static_features=static,
        template=template,
        initial=initial,
        aggregate_names=datamodule.aggregate_names(cfg),
        extras={"train_steps": steps, "seed": cfg.seed, "profile": cfg.profile_name},
    )
    summary = out.parent / "train_summary.json"
    summary.write_text(json.dumps({"metrics": metrics, "history": history}, indent=2))
    log.info("%s trained: val_total %.4f -> %s", arch, metrics["val_total"], out)
    return TrainResult(arch=arch, checkpoint=out, summary=summary, metrics=metrics)
