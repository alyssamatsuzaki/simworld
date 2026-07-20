"""§11 family 9 — out-of-distribution honesty, reported without spin.

Two probes: (a) held-out emulator error against the Mahalanobis distance of
each episode's action sequence from the training action distribution; (b)
enforcement pushed to 1.5, fifty percent beyond the training range, graded
against fresh tensorized-ABM truth.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from regworld import rules
from regworld.abm.model import load_observed_world
from regworld.abm.tensorized import rollout_tensorized
from regworld.evaluation.harness import EvalContext, open_loop_natural
from regworld.training.datamodule import EmulatorSequences, load_theta_draws


def _mahalanobis(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
    d = x - mean
    return float(np.sqrt(d @ cov_inv @ d))


def evaluate(ctx: EvalContext) -> dict[str, object]:
    cfg = ctx.cfg
    train = EmulatorSequences(cfg, "train")
    train_actions = np.stack(
        [train.episode_arrays(e)["action"][1:].mean(axis=0) for e in train.episodes]
    )
    mean = train_actions.mean(axis=0)
    cov = np.cov(train_actions.T) + 1e-6 * np.eye(4)
    cov_inv = np.linalg.inv(cov)

    batch = ctx.batch
    horizon = batch["firm"].shape[1] - 1
    agg, _, _ = open_loop_natural(
        ctx.model,
        batch,
        burn_in=1,
        horizon=horizon,
        generator=torch.Generator().manual_seed(cfg.seed + 44_000),
    )
    errors = np.abs(np.clip(agg[..., 0], 0, 1) - batch["aggregate"][:, 1:, 0].numpy()).mean(axis=1)
    distances = np.array(
        [
            _mahalanobis(batch["action"][i, 1:].numpy().mean(axis=0), mean, cov_inv)
            for i in range(batch["action"].shape[0])
        ]
    )
    corr = float(stats.spearmanr(distances, errors).statistic) if len(errors) > 2 else None

    # Enforcement at 1.5: outside the [0, 1] training range by half the range.
    world = load_observed_world(cfg)
    theta_rows = load_theta_draws(cfg)
    names = list(rules.Theta.__dataclass_fields__)
    theta = rules.Theta(**dict(zip(names, theta_rows.mean(axis=0).tolist(), strict=True)))
    quarters = cfg.horizon_quarters
    extreme = np.tile(np.array([1.5, 0.0, 0.5, 0.0]), (quarters, 1))
    truth_run = rollout_tensorized(
        cfg,
        world,
        theta,
        rules.PolicyLevers(),
        seed=cfg.seed + 45_000,
        quarters=quarters,
        lever_schedule=extreme,
    )
    truth_compliance = np.array([float(o.compliance_rate.item()) for o in truth_run.outcomes])
    initial = ctx.heldout.initial_arrays()
    ood_batch = {
        "firm": torch.as_tensor(initial["firm"], dtype=torch.float32)[None, None].expand(
            -1, quarters + 1, -1, -1
        ),
        "segment": torch.as_tensor(initial["segment"], dtype=torch.float32)[None, None].expand(
            -1, quarters + 1, -1, -1
        ),
        "aggregate": torch.as_tensor(initial["aggregate"], dtype=torch.float32)[None, None].expand(
            -1, quarters + 1, -1
        ),
        "action": torch.cat(
            [torch.zeros(1, 1, 4), torch.as_tensor(extreme, dtype=torch.float32)[None]], dim=1
        ),
    }
    with torch.no_grad():
        ood_agg, _, _ = open_loop_natural(
            ctx.model,
            {k: v.clone() for k, v in ood_batch.items()},
            burn_in=1,
            horizon=quarters,
            generator=torch.Generator().manual_seed(cfg.seed + 46_000),
        )
    ood_error = float(np.abs(np.clip(ood_agg[0, :, 0], 0, 1) - truth_compliance).mean())
    in_range_error = float(errors.mean())
    extreme_distance = _mahalanobis(extreme.mean(axis=0), mean, cov_inv)
    return {
        "heldout_error_vs_mahalanobis_spearman": corr,
        "heldout_mean_error": in_range_error,
        "heldout_distances": [round(float(d), 2) for d in distances],
        "enforcement_1p5_error": ood_error,
        "enforcement_1p5_mahalanobis": round(extreme_distance, 2),
        "error_growth_factor_at_1p5": round(ood_error / max(in_range_error, 1e-9), 2),
        "note": "reported without spin (§11 Fig 13): expect error to grow with distance",
    }
