"""§11 family 1 — predictive accuracy: 1-step AUC/MAE and k-step rollout drift.

The useful-range statement is required in plain words: the horizon at which
compliance error crosses 0.10 is where imagination stops being informative.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from regworld.evaluation.harness import EvalContext, open_loop_natural


def one_step_metrics(ctx: EvalContext) -> dict[str, float]:
    """Prior-step predictions from every posterior frame, pooled."""
    batch = ctx.batch
    steps = batch["firm"].shape[1]
    probs, targets, alive_masks, agg_errors = [], [], [], []
    for t in range(1, steps):
        agg, node_probs, start = open_loop_natural(
            ctx.model,
            batch,
            burn_in=t,
            horizon=1,
            generator=torch.Generator().manual_seed(1000 + t),
        )
        assert start == t - 1
        probs.append(node_probs[:, 0])
        targets.append(batch["firm"][:, t, :, 0].numpy())
        alive_masks.append(batch["firm"][:, t, :, 1].numpy() > 0.5)
        agg_errors.append(np.abs(agg[:, 0] - batch["aggregate"][:, t].numpy()))
    probs_arr = np.concatenate([p[m] for p, m in zip(probs, alive_masks, strict=True)])
    # The compliance label is a bit; the straight-through Bernoulli leaves a float32
    # artifact (0.99999994) in storage, so threshold before AUC wants binary labels.
    targets_arr = (
        np.concatenate([t[m] for t, m in zip(targets, alive_masks, strict=True)]) > 0.5
    ).astype(np.int64)
    auc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 1.0
    errors = np.stack(agg_errors)  # (T-1, B, A)
    return {
        "one_step_node_auc": auc,
        "one_step_compliance_mae": float(errors[..., 0].mean()),
        "one_step_hhi_mae": float(errors[..., 2].mean()),
        "one_step_aggregate_mae": float(errors.mean()),
    }


def k_step_drift(ctx: EvalContext) -> dict[str, object]:
    """Open-loop drift from the initial frame, against persistence."""
    batch = ctx.batch
    steps = batch["firm"].shape[1]
    horizon = steps - 1
    agg, _, start = open_loop_natural(
        ctx.model,
        batch,
        burn_in=1,
        horizon=horizon,
        generator=torch.Generator().manual_seed(7),
    )
    target = batch["aggregate"][:, start + 1 : start + 1 + horizon].numpy()
    persistence = batch["aggregate"][:, start : start + 1].numpy()
    drift: dict[str, float] = {}
    persistence_drift: dict[str, float] = {}
    ks = [k for k in ctx.cfg.eval.k_steps if k <= horizon]
    for k in ks:
        drift[f"k{k}"] = float(np.abs(agg[:, k - 1, 0] - target[:, k - 1, 0]).mean())
        persistence_drift[f"k{k}"] = float(
            np.abs(persistence[:, 0, 0] - target[:, k - 1, 0]).mean()
        )
    per_step = np.abs(agg[..., 0] - target[..., 0]).mean(axis=0)  # (K,)
    exceeded = np.flatnonzero(per_step > 0.10)
    useful_range = int(exceeded[0]) if exceeded.size else horizon
    beats = sum(drift[f"k{k}"] < persistence_drift[f"k{k}"] for k in ks)
    return {
        "compliance_drift": drift,
        "persistence_drift": persistence_drift,
        "beats_persistence_at": f"{beats}/{len(ks)} horizons",
        "useful_range_quarters": useful_range,
        "useful_range_statement": (
            f"Compliance error stays below 0.10 for {useful_range} quarters of pure "
            f"imagination; beyond that the emulator should hand back to the ABM."
        ),
        "final_step_mae": float(per_step[-1]),
    }


def evaluate(ctx: EvalContext) -> dict[str, object]:
    out: dict[str, object] = {}
    out.update(one_step_metrics(ctx))
    out.update(k_step_drift(ctx))
    out["thresholds_dev"] = {
        "one_step_node_auc": ">= 0.85",
        "one_step_compliance_mae": "<= 0.02",
        "q24_compliance_mae": "<= 0.06",
    }
    return out
