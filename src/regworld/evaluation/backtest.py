"""§11 family 10 — historical backtest: condition on q1-12, predict q13-24.

No peeking: the model filters the first twelve quarters of each held-out
trajectory (posterior), then rolls the prior for the remaining twelve; coverage
of the 90% predictive interval on the held-out window is the graded number.
"""

from __future__ import annotations

import numpy as np

from regworld.evaluation.harness import EvalContext, sampled_open_loop


def evaluate(ctx: EvalContext, n_samples: int = 16) -> dict[str, object]:
    batch = ctx.batch
    steps = batch["firm"].shape[1]
    burn_in = min(13, steps - 2)  # frames 0..12 = the observed regime window
    horizon = steps - burn_in
    samples = sampled_open_loop(
        ctx.model,
        batch,
        burn_in=burn_in,
        horizon=horizon,
        n_samples=n_samples,
        base_seed=ctx.cfg.seed + 47_000,
    )  # (S, B, K, A)
    compliance = np.clip(samples[..., 0], 0.0, 1.0)
    truth = batch["aggregate"][:, burn_in : burn_in + horizon, 0].numpy()
    lo = np.quantile(compliance, 0.05, axis=0)
    hi = np.quantile(compliance, 0.95, axis=0)
    coverage = float(np.mean((truth >= lo) & (truth <= hi)))
    mae = float(np.abs(compliance.mean(axis=0) - truth).mean())
    return {
        "conditioning_quarters": burn_in - 1,
        "held_out_quarters": horizon,
        "coverage_90_heldout_window": coverage,
        "mean_prediction_mae": mae,
        "n_samples": n_samples,
        "thresholds_dev": {"coverage_90_heldout_window": "in [0.85, 0.95]"},
    }
