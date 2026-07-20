"""§11 family 6 — stylized facts never fit during calibration.

Checked on emulator-imagined trajectories against the ABM corpus: S-shaped
adoption under strong enforcement, the compliance-by-size gradient, the
exit-vs-enforcement relationship, and the heavy-tailed firm-size distribution
(a world property, reported for context). Cascade-size tails are deferred to
the Phase 7 report — they need event-level cascade extraction that Phase 5
does not produce.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import optimize, stats

from regworld.abm.model import load_observed_world
from regworld.evaluation.harness import EvalContext, open_loop_natural
from regworld.training.datamodule import EmulatorSequences


def _logistic(t: np.ndarray, level: float, rate: float, midpoint: float) -> np.ndarray:
    return level / (1.0 + np.exp(-rate * (t - midpoint)))


def s_curve_r2(trajectory: np.ndarray) -> float:
    """R^2 of the best-fit logistic to one compliance trajectory."""
    t = np.arange(len(trajectory), dtype=np.float64)
    try:
        popt, _ = optimize.curve_fit(
            _logistic,
            t,
            trajectory,
            p0=[max(trajectory.max(), 0.1), 0.5, len(t) / 3],
            maxfev=5000,
        )
    except RuntimeError:
        return 0.0
    residual = trajectory - _logistic(t, *popt)
    total = trajectory - trajectory.mean()
    denom = float(np.sum(total**2))
    return 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else 0.0


def evaluate(ctx: EvalContext) -> dict[str, object]:
    cfg = ctx.cfg
    batch = ctx.batch
    horizon = batch["firm"].shape[1] - 1
    agg, _, _ = open_loop_natural(
        ctx.model,
        batch,
        burn_in=1,
        horizon=horizon,
        generator=torch.Generator().manual_seed(cfg.seed + 43_000),
    )
    imagined_compliance = np.clip(agg[..., 0], 0.0, 1.0)  # (B, K)
    real_compliance = batch["aggregate"][:, 1:, 0].numpy()
    actions = batch["action"][:, 1:, 0].numpy()  # enforcement lever
    mean_enforcement = actions.mean(axis=1)

    # S-curve: strong-enforcement episodes should adopt logistically.
    strong = mean_enforcement > 0.5
    if strong.sum() == 0:
        strong = mean_enforcement >= np.median(mean_enforcement)
    emu_r2 = [s_curve_r2(imagined_compliance[i]) for i in np.flatnonzero(strong)]
    abm_r2 = [s_curve_r2(real_compliance[i]) for i in np.flatnonzero(strong)]

    # Compliance-by-size gradient at the terminal quarter: slope over deciles.
    n_sectors = cfg.population.n_sectors
    decile_slice = slice(8 + n_sectors, 8 + n_sectors + 10)
    emu_deciles = np.clip(agg[:, -1, decile_slice], 0.0, 1.0).mean(axis=0)
    abm_deciles = batch["aggregate"][:, -1, decile_slice].numpy().mean(axis=0)
    decile_index = np.arange(10, dtype=np.float64)
    emu_gradient = float(np.polyfit(decile_index, emu_deciles, 1)[0])
    abm_gradient = float(np.polyfit(decile_index, abm_deciles, 1)[0])

    # Exit-vs-enforcement across the full corpus (train + heldout real data),
    # and the same relationship inside the emulator's imagination.
    corpus = EmulatorSequences(cfg, "train")
    exits, enforce = [], []
    for episode in corpus.episodes:
        arrays = corpus.episode_arrays(episode)
        exits.append(float(arrays["aggregate"][-1, 5]))
        enforce.append(float(arrays["action"][1:, 0].mean()))
    abm_exit_corr = float(stats.spearmanr(enforce, exits).statistic) if len(exits) > 2 else 0.0
    emu_exit = np.clip(agg[:, -1, 5], 0.0, 1.0)
    emu_exit_corr = (
        float(stats.spearmanr(mean_enforcement, emu_exit).statistic) if len(emu_exit) > 2 else 0.0
    )

    # Firm sizes: heavy-tailed by construction; report the log-normal sigma.
    world = load_observed_world(cfg)
    log_sizes = np.log(world.firms.size[world.firms.size > 0])

    return {
        "s_curve_r2_emulator_mean": float(np.mean(emu_r2)) if emu_r2 else None,
        "s_curve_r2_abm_mean": float(np.mean(abm_r2)) if abm_r2 else None,
        "n_strong_enforcement_episodes": int(strong.sum()),
        "size_gradient_emulator": emu_gradient,
        "size_gradient_abm": abm_gradient,
        "size_gradient_sign_match": bool(np.sign(emu_gradient) == np.sign(abm_gradient)),
        "exit_vs_enforcement_corr_abm": abm_exit_corr,
        "exit_vs_enforcement_corr_emulator": emu_exit_corr,
        "exit_corr_sign_match": bool(np.sign(abm_exit_corr) == np.sign(emu_exit_corr))
        if abm_exit_corr != 0.0
        else None,
        "firm_size_lognormal_sigma": float(log_sizes.std()),
        "cascade_sizes": "deferred to Phase 7 (needs event-level cascade extraction)",
        "thresholds_dev": {"s_curve_r2": ">= 0.9", "gradients": "correct sign"},
    }
