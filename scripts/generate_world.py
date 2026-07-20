"""Stage 1a: generate the ground-truth world → observed/ + oracle/ (world builder)."""

import hydra
from omegaconf import DictConfig

from regworld.data.generate import generate_ground_truth
from regworld.logging_conf import setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    generate_ground_truth(cfg_obj)


if __name__ == "__main__":
    main()
