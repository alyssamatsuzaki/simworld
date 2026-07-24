"""Stage 10 (Phase 6): SB3 PPO (control group) + latent Dreamer-style
actor-critic (experiment) trained against the emulator (`make emulator`
first). Mirrors `scripts/train_emulator.py`.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from simworld.agents import train_rl
from simworld.logging_conf import get_logger, setup_logging
from simworld.seeding import seed_everything
from simworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    result = train_rl(cfg_obj)
    log.info(
        "Stage 10 done: %d checkpoint(s) -> %s",
        len(result.checkpoints),
        result.summary,
    )


if __name__ == "__main__":
    main()
