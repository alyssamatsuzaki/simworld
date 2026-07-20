"""Stages 6+7: train the GraphRSSM on the domain-randomized corpus.

Collects the dataset first if it is missing, so `make emulator` is
self-contained after Stages 1-4 have run.
"""

import hydra
from omegaconf import DictConfig

from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.training.train_emulator import train_world_model
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    result = train_world_model(cfg_obj)
    log.info(
        "emulator %s trained: val_total=%.4f checkpoint=%s",
        result.arch,
        result.metrics["val_total"],
        result.checkpoint,
    )


if __name__ == "__main__":
    main()
