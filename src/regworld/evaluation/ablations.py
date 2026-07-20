"""§11 family 11 — ablations: if removing a component doesn't hurt, it was
decoration, and the report says so.

Phase-5 scope trains the three arches (`rssm_gnn`, `rssm_flat`, `gru_baseline`)
on the shared corpus with identical budgets and compares held-out one-step MAE
and open-loop drift. The remaining §11 ablation axes (Gaussian latents,
node-level stochasticity, no-free-bits, prior-draw calibration) are declared
and deferred to the dev-profile run with >= 3 seeds.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from regworld.evaluation import harness
from regworld.training.checkpoint import checkpoint_path
from regworld.training.train_emulator import train_world_model
from regworld.types import RegWorldConfig

ARCHES = ("rssm_gnn", "rssm_flat", "gru_baseline")


def evaluate(cfg: RegWorldConfig) -> dict[str, object]:
    rows = []
    for arch in ARCHES:
        path = checkpoint_path(cfg.paths.root, arch)
        if not path.is_file():
            train_world_model(cfg, arch=arch)
        summary = json.loads((path.parent / "train_summary.json").read_text())
        ctx = harness.load_context(cfg, arch=arch)
        batch = ctx.batch
        horizon = batch["firm"].shape[1] - 1
        agg, _, _ = harness.open_loop_natural(
            ctx.model,
            batch,
            burn_in=1,
            horizon=horizon,
            generator=torch.Generator().manual_seed(cfg.seed + 48_000),
        )
        drift = float(
            np.abs(np.clip(agg[..., 0], 0, 1) - batch["aggregate"][:, 1:, 0].numpy()).mean()
        )
        agg1, _, _ = harness.open_loop_natural(
            ctx.model,
            batch,
            burn_in=horizon // 2,
            horizon=1,
            generator=torch.Generator().manual_seed(cfg.seed + 49_000),
        )
        one_step = float(
            np.abs(
                np.clip(agg1[:, 0, 0], 0, 1) - batch["aggregate"][:, horizon // 2, 0].numpy()
            ).mean()
        )
        rows.append(
            {
                "arch": arch,
                "val_total": round(summary["metrics"]["val_total"], 4),
                "one_step_compliance_mae": round(one_step, 4),
                "open_loop_compliance_mae": round(drift, 4),
                "parameters": int(summary["metrics"]["parameters"]),
            }
        )
    by_drift = sorted(rows, key=lambda r: r["open_loop_compliance_mae"])
    gnn = next(r for r in rows if r["arch"] == "rssm_gnn")
    flat = next(r for r in rows if r["arch"] == "rssm_flat")
    return {
        "table": rows,
        "best_by_open_loop": by_drift[0]["arch"],
        "gnn_beats_flat": gnn["open_loop_compliance_mae"] < flat["open_loop_compliance_mae"],
        "verdict_note": (
            "if rssm_gnn does not beat rssm_flat, the graph structure was decoration "
            "and the report says so (§10 Stages 6+7)"
        ),
        "deferred_axes": [
            "discrete vs Gaussian latents",
            "node-level stochastic latents",
            "with/without KL free bits",
            "no-calibration prior draws",
            ">= 3 seeds per cell (dev profile)",
        ],
    }
