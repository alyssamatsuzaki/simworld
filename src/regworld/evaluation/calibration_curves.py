"""§11 family 3 — calibration: interval coverage, ECE, reliability curve.

Do 90% predictive intervals contain reality 90% of the time? Coverage comes
from the stochastic-latent ensemble on held-out trajectories; ECE from pooled
one-step node compliance probabilities.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from regworld.evaluation.harness import EvalContext, open_loop_natural, sampled_open_loop

LEVELS = (0.50, 0.80, 0.90, 0.95)


def interval_coverage(ctx: EvalContext, n_samples: int = 16) -> dict[str, float]:
    batch = ctx.batch
    horizon = batch["firm"].shape[1] - 1
    samples = sampled_open_loop(
        ctx.model,
        batch,
        burn_in=1,
        horizon=horizon,
        n_samples=n_samples,
        base_seed=ctx.cfg.seed + 41_000,
    )  # (S, B, K, A)
    compliance = np.clip(samples[..., 0], 0.0, 1.0)
    truth = batch["aggregate"][:, 1 : 1 + horizon, 0].numpy()  # (B, K)
    out: dict[str, float] = {}
    for level in LEVELS:
        lo = np.quantile(compliance, (1 - level) / 2, axis=0)
        hi = np.quantile(compliance, 1 - (1 - level) / 2, axis=0)
        out[f"coverage_{int(level * 100)}"] = float(np.mean((truth >= lo) & (truth <= hi)))
    return out


def node_ece(ctx: EvalContext, n_bins: int = 10, stride: int = 2) -> dict[str, object]:
    """Expected calibration error of one-step firm compliance probabilities."""
    batch = ctx.batch
    steps = batch["firm"].shape[1]
    probs, targets = [], []
    for t in range(1, steps, stride):
        _, node_probs, _ = open_loop_natural(
            ctx.model,
            batch,
            burn_in=t,
            horizon=1,
            generator=torch.Generator().manual_seed(2000 + t),
        )
        alive = batch["firm"][:, t, :, 1].numpy() > 0.5
        probs.append(node_probs[:, 0][alive])
        targets.append(batch["firm"][:, t, :, 0].numpy()[alive])
    p = np.concatenate(probs)
    y = np.concatenate(targets)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability: list[dict[str, object]] = []
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            reliability.append(
                {"bin": f"[{lo:.1f},{hi:.1f})", "confidence": None, "accuracy": None}
            )
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (mask.sum() / len(p)) * abs(confidence - accuracy)
        reliability.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "confidence": round(confidence, 4),
                "accuracy": round(accuracy, 4),
                "count": int(mask.sum()),
            }
        )
    return {"ece": float(ece), "reliability_diagram": reliability}


def evaluate(ctx: EvalContext) -> dict[str, object]:
    out: dict[str, object] = {}
    out.update(interval_coverage(ctx))
    out.update(node_ece(ctx))
    out["thresholds_dev"] = {"coverage_90": "in [0.85, 0.95]", "ece": "<= 0.05"}
    return out
