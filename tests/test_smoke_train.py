"""The single best deep-learning unit test there is (§10 Stages 6+7):

overfit one batch — the reconstruction loss must fall below 0.05x its initial
value within 200 steps — and the open-loop rollout must beat the persistence
("no change") baseline on that batch. Held-out-episode persistence comparison
runs in `scripts/eval_emulator.py` on the real corpus; this stays synthetic so
the fast suite needs no artifacts.
"""

from __future__ import annotations

import torch

from regworld.training.losses import (
    symlog,
    symlog_mse,
    two_hot_bins,
    two_hot_decode,
    two_hot_encode,
)
from regworld.training.train_emulator import reconstruction_loss, train_step

from .test_dynamics_shapes import tiny_batch, tiny_model


def _recon(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        losses = model.observe_losses(batch)
        losses.update(model.imagination_losses(batch, burn_in=2, horizon=3))
    return reconstruction_loss(losses)


def test_overfit_one_batch_and_beat_persistence() -> None:
    torch.manual_seed(0)
    model = tiny_model("rssm_gnn", seed=1)
    batch = tiny_batch(seed=1, deterministic_targets=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    initial = _recon(model, batch)
    for _ in range(200):
        train_step(model, batch, optimizer, burn_in=2, horizon=3, grad_clip=100.0)
    final = _recon(model, batch)
    assert final < 0.05 * initial, (
        f"overfit-one-batch failed: recon {final:.4f} vs initial {initial:.4f} "
        f"(ratio {final / initial:.3f} >= 0.05)"
    )

    # k-step open loop must beat persistence ("no change from the burn-in frame").
    model.eval()
    with torch.no_grad():
        agg_pred, _, start = model.open_loop(batch, burn_in=2, horizon=3)
    k = agg_pred.shape[1]
    target = batch["aggregate"][:, start + 1 : start + 1 + k]
    persistence = batch["aggregate"][:, start : start + 1].expand_as(target)
    model_err = float(symlog_mse(agg_pred, target))
    persistence_err = float(torch.mean((symlog(persistence) - symlog(target)) ** 2))
    assert model_err < persistence_err, (
        f"open-loop rollout ({model_err:.5f}) does not beat persistence ({persistence_err:.5f})"
    )


def test_two_hot_round_trip() -> None:
    bins = two_hot_bins(63)
    values = torch.tensor([-2.0, -0.3, 0.0, 0.7, 4.0])
    encoding = two_hot_encode(values, bins)
    assert torch.allclose(encoding.sum(-1), torch.ones(5))
    decoded = two_hot_decode(torch.log(encoding.clamp_min(1e-8)), bins)
    assert torch.allclose(decoded, values, atol=0.05)


def test_training_loop_runs_without_nan() -> None:
    """50-step train without NaN (§12 numerical class), all three arches."""
    for arch in ("rssm_gnn", "rssm_flat", "gru_baseline"):
        model = tiny_model(arch, seed=2)
        batch = tiny_batch(seed=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(50):
            losses = train_step(model, batch, optimizer, burn_in=2, horizon=3, grad_clip=100.0)
        assert all(torch.isfinite(v) for v in losses.values()), arch
