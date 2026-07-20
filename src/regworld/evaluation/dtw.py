"""§11 family 4 — trajectory shape via dynamic time warping (~30 lines, no dep).

Imagined vs real compliance trajectories, against a shuffled-pairing baseline.
Reported, not gated.
"""

from __future__ import annotations

import numpy as np
import torch

from regworld.evaluation.harness import EvalContext, open_loop_natural


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(nm) DTW with absolute-difference local cost."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def evaluate(ctx: EvalContext) -> dict[str, object]:
    batch = ctx.batch
    horizon = batch["firm"].shape[1] - 1
    agg, _, _ = open_loop_natural(
        ctx.model,
        batch,
        burn_in=1,
        horizon=horizon,
        generator=torch.Generator().manual_seed(ctx.cfg.seed + 42_000),
    )
    imagined = np.clip(agg[..., 0], 0.0, 1.0)  # (B, K)
    real = batch["aggregate"][:, 1 : 1 + horizon, 0].numpy()
    matched = [dtw_distance(imagined[i], real[i]) for i in range(real.shape[0])]
    # Baseline: imagined trajectory of episode i against the real trajectory of
    # a different episode — how much of the match is episode-specific signal?
    shuffled = [
        dtw_distance(imagined[i], real[(i + 1) % real.shape[0]]) for i in range(real.shape[0])
    ]
    return {
        "dtw_matched_mean": float(np.mean(matched)),
        "dtw_shuffled_baseline_mean": float(np.mean(shuffled)),
        "matched_beats_shuffled": bool(np.mean(matched) < np.mean(shuffled)),
        "note": "reported, not gated (§11 family 4)",
    }
