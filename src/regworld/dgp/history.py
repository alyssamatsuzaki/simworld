"""Regime P: the analogous prior regulation with a staggered regional rollout (§7.8).

Enforcement switches on region by region at quarters t_r drawn INDEPENDENTLY of firm
characteristics — exogenous by construction, which is what identifies the DiD. The
policy levers of the past regime are the status-quo `phased_targeted` schedule.
"""

from __future__ import annotations

import numpy as np

from regworld.dgp.dynamics import Trajectory, run_dgp
from regworld.rules import FirmAttributes, Graphs, PolicyLevers, SegmentAttributes
from regworld.types import RegWorldConfig

# The past regime's levers: the status-quo policy (§10 Stage 10a).
REGIME_P_LEVERS = PolicyLevers(enforcement=0.6, targeting=0.5, phase_speed=0.3, subsidy=0.3)

# Rollout quarters are drawn from this window (0-based): at least 2 pre-treatment
# quarters for every region and onset inside the q1-12 observation window.
ROLLOUT_EARLIEST = 2
ROLLOUT_LATEST = 9

NEVER_TREATED = 10_000  # t_start sentinel: enforcement never arrives


def draw_rollout(cfg: RegWorldConfig, rng: np.random.Generator) -> np.ndarray:
    """t_r per region, uniform on the rollout window, independent of everything."""
    return rng.integers(ROLLOUT_EARLIEST, ROLLOUT_LATEST + 1, size=cfg.population.n_regions)


def run_history(
    cfg: RegWorldConfig,
    firms: FirmAttributes,
    segments: SegmentAttributes,
    graphs: Graphs,
    seed: int,
    *,
    rollout: np.ndarray | None = None,
    force_all_treated: bool = False,
    force_never_treated: bool = False,
) -> tuple[Trajectory, np.ndarray]:
    """24 quarters of Regime P. Returns (trajectory, per-firm treatment quarter).

    The force_* flags exist for the ground-truth do() runs (§7.10): identical seed and
    entities, counterfactual rollout.
    """
    rng = np.random.default_rng(seed + 90_001)  # rollout stream, distinct from dynamics
    t_r = rollout if rollout is not None else draw_rollout(cfg, rng)
    t_start = t_r[firms.region].astype(np.int64)
    if force_all_treated:
        t_start = np.zeros_like(t_start)
    if force_never_treated:
        t_start = np.full_like(t_start, NEVER_TREATED)
    traj = run_dgp(
        cfg,
        firms,
        segments,
        graphs,
        REGIME_P_LEVERS,
        seed,
        cfg.horizon_quarters,
        t_start=t_start,
    )
    return traj, t_start
