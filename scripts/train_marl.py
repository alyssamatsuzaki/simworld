"""Stage 10d (Phase 6): strategic-firm MARL by iterated best response, and the
claim-C6 comparison it makes computable.

DEGRADED by design — RLlib is non-gating (PLAN.md guardrail 11), so this runs
the plan's sanctioned fallback: independent PPO over single-agent views of the
Stage-9 PettingZoo env with the other agents frozen, iterated over rounds.
Writes ``artifacts/marl/c6_comparison.json``. Mirrors ``scripts/train_rl.py``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from regworld.agents.marl import train_marl
from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    result = train_marl(cfg_obj)
    log.info(
        "Stage 10d done: %d checkpoint(s), C6 comparison -> %s",
        len(result.checkpoints),
        result.comparison,
    )


if __name__ == "__main__":
    main()
