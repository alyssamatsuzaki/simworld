"""Stage 3: run the observed-world Mesa ABM and write durable outputs."""

import logging

import hydra
from omegaconf import DictConfig

from regworld.abm.collect import run_observed_abm
from regworld.logging_conf import setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    trajectory, paths = run_observed_abm(cfg_obj)
    terminal = trajectory.outcomes[-1]
    log.info(
        "ABM complete: %d quarters, terminal compliance %.3f, exit rate %.3f; %d outputs",
        len(trajectory.outcomes),
        terminal.compliance_rate,
        terminal.exit_rate_cum,
        len(paths),
    )


if __name__ == "__main__":
    main()
