"""Stage 6+7 training data: domain-randomized tensorized-ABM rollouts -> Zarr."""

import hydra
from omegaconf import DictConfig

from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.training.datamodule import build_dataset
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    out = build_dataset(cfg_obj)
    log.info("emulator dataset written to %s", out)


if __name__ == "__main__":
    main()
