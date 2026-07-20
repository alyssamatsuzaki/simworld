"""§11 family 2 — distributional fidelity on terminal outcomes.

Wasserstein-1, RBF-MMD, energy distance, a permutation test (we want to FAIL to
distinguish emulator from ABM), and the NLL of ABM terminals under a KDE of the
emulator's ensemble. Both ensembles run the historical ``phased_targeted``
policy with theta drawn from the Stage-4 posterior.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from regworld import rules
from regworld.abm.model import load_observed_world
from regworld.abm.policies import STATIC_POLICIES
from regworld.abm.tensorized import rollout_tensorized
from regworld.evaluation.harness import EvalContext
from regworld.training.datamodule import load_theta_draws
from regworld.training.losses import symexp


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """RBF-kernel MMD^2 with the median-distance bandwidth heuristic."""
    z = np.concatenate([x, y])[:, None]
    d2 = (z - z.T) ** 2
    bandwidth = np.median(d2[d2 > 0]) or 1.0
    k = np.exp(-d2 / bandwidth)
    n = len(x)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean())


def _permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = stats.wasserstein_distance(x, y)
    pooled = np.concatenate([x, y])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if stats.wasserstein_distance(pooled[: len(x)], pooled[len(x) :]) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def evaluate(ctx: EvalContext) -> dict[str, object]:
    cfg = ctx.cfg
    n_rollouts = cfg.eval.n_dist_rollouts
    quarters = cfg.horizon_quarters
    world = load_observed_world(cfg)
    theta_rows = load_theta_draws(cfg)
    names = list(rules.Theta.__dataclass_fields__)
    levers = STATIC_POLICIES["phased_targeted"]
    schedule = np.tile(levers.as_array(), (quarters, 1))
    rng = np.random.default_rng(cfg.seed + 95_000)

    abm_compliance, abm_hhi = [], []
    for k in range(n_rollouts):
        draw = theta_rows[int(rng.integers(theta_rows.shape[0]))]
        theta = rules.Theta(**dict(zip(names, draw.tolist(), strict=True)))
        run = rollout_tensorized(
            cfg, world, theta, levers, seed=cfg.seed + 96_000 + k, quarters=quarters
        )
        terminal = run.outcomes[-1]
        abm_compliance.append(float(terminal.compliance_rate.item()))
        abm_hhi.append(float(terminal.hhi.item()))

    initial = ctx.heldout.initial_arrays()
    batch = {
        "firm": torch.as_tensor(initial["firm"], dtype=torch.float32)[None, None],
        "segment": torch.as_tensor(initial["segment"], dtype=torch.float32)[None, None],
        "aggregate": torch.as_tensor(initial["aggregate"], dtype=torch.float32)[None, None],
        "action": torch.zeros(1, 1, 4),
    }
    actions = torch.as_tensor(schedule, dtype=torch.float32)[None]
    full_actions = torch.cat([batch["action"], actions], dim=1)
    rollout_batch = {
        "firm": batch["firm"].expand(-1, quarters + 1, -1, -1).clone(),
        "segment": batch["segment"].expand(-1, quarters + 1, -1, -1).clone(),
        "aggregate": batch["aggregate"].expand(-1, quarters + 1, -1).clone(),
        "action": full_actions,
    }
    emu_compliance, emu_hhi = [], []
    with torch.no_grad():
        for s in range(n_rollouts):
            generator = torch.Generator().manual_seed(cfg.seed + 97_000 + s)
            agg_symlog, _, _ = ctx.model.open_loop(
                rollout_batch, burn_in=1, horizon=quarters, generator=generator
            )
            terminal_vec = symexp(agg_symlog[0, -1]).numpy()
            emu_compliance.append(float(np.clip(terminal_vec[0], 0.0, 1.0)))
            emu_hhi.append(float(np.clip(terminal_vec[2], 0.0, 10_000.0)))

    abm_c, emu_c = np.asarray(abm_compliance), np.asarray(emu_compliance)
    abm_h, emu_h = np.asarray(abm_hhi), np.asarray(emu_hhi)
    kde_c = stats.gaussian_kde(emu_c + rng.normal(0, 1e-4, emu_c.shape))
    return {
        "n_rollouts_per_side": n_rollouts,
        "w1_compliance": float(stats.wasserstein_distance(abm_c, emu_c)),
        "w1_hhi": float(stats.wasserstein_distance(abm_h, emu_h)),
        "w1_hhi_fraction_of_range": float(stats.wasserstein_distance(abm_h, emu_h)) / 10_000.0,
        "mmd_compliance": _mmd_rbf(abm_c, emu_c),
        "energy_compliance": float(stats.energy_distance(abm_c, emu_c)),
        "permutation_p_compliance": _permutation_p(abm_c, emu_c, seed=cfg.seed),
        "nll_abm_under_emulator_kde": float(-np.mean(kde_c.logpdf(abm_c))),
        "abm_terminal_compliance_mean": float(abm_c.mean()),
        "emulator_terminal_compliance_mean": float(emu_c.mean()),
        "abm_terminal_compliance": [round(float(v), 4) for v in abm_c],
        "emulator_terminal_compliance": [round(float(v), 4) for v in emu_c],
        "thresholds_dev": {
            "w1_compliance": "<= 0.03",
            "w1_hhi": "<= 0.01 (as a fraction of the 10,000 HHI range)",
            "permutation_test": "must NOT reject at p < 0.01",
        },
    }
