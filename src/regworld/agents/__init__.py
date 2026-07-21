"""Stage 10 (Phase 6): policy learning against the trained emulator.

Everything the regulator can be under evaluation lives here: scripted
baselines (``abm.policies.STATIC_POLICIES``, reused rather than duplicated),
SB3 PPO trained inside ``EmulatorEnv`` (:mod:`regworld.agents.ppo`, the
control group), and a latent Dreamer-style actor-critic trained purely on
imagined rollouts (:mod:`regworld.agents.dreamer`, the experiment).
:mod:`regworld.agents.registry` is the single lookup both this stage and the
Stage-11 ensemble use to turn a policy name into a callable, and
:mod:`regworld.agents.planning` rolls any such callable out in either world.

:func:`train_rl` is the package's single entry point: it loads the emulator
checkpoint once, trains PPO, optionally trains the Dreamer agent, and writes
one run summary. Both ``scripts/train_rl.py`` and the driver's ``stage_rl``
call it.

This package must never import :mod:`regworld.dgp` or read from the
answer-key artifact tree — a dedicated grep test enforces it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from regworld.agents.dreamer import train_dreamer
from regworld.agents.ppo import train_ppo
from regworld.agents.registry import load_checkpoint_compat
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

__all__ = ["RlResult", "train_rl"]


@dataclass
class RlResult:
    checkpoints: list[Path]
    summary: Path
    metrics: dict[str, float] = field(default_factory=dict)


def train_rl(cfg: RegWorldConfig) -> RlResult:
    """Train every Stage-10 policy against the emulator and write the run summary."""
    model, meta = load_checkpoint_compat(cfg)
    model = model.eval()

    checkpoints: list[Path] = []
    metrics: dict[str, float] = {}

    ppo_result = train_ppo(cfg, model=model, meta=meta)
    checkpoints.append(ppo_result.checkpoint)
    metrics.update({f"ppo_{key}": value for key, value in ppo_result.metrics.items()})

    if cfg.rl.train_dreamer:
        dreamer_result = train_dreamer(cfg, model=model, meta=meta)
        checkpoints.extend([dreamer_result.actor_path, dreamer_result.meta_path])
        metrics.update({f"dreamer_{key}": value for key, value in dreamer_result.metrics.items()})
    else:
        log.info("cfg.rl.train_dreamer is False; skipping the Stage 10c latent agent")

    out_dir = Path(cfg.paths.root) / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "rl_summary.json"
    summary.write_text(json.dumps({"profile": cfg.profile_name, "metrics": metrics}, indent=2))
    log.info("Stage 10 policies trained: %d checkpoint(s) -> %s", len(checkpoints), summary)
    return RlResult(checkpoints=checkpoints, summary=summary, metrics=metrics)
