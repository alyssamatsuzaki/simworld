"""Stage 1b: observed tables → analysis-ready panel + DuckDB views."""

import hydra
from omegaconf import DictConfig

from simworld.data.duck import build_views
from simworld.data.ingest import ingest
from simworld.logging_conf import setup_logging
from simworld.seeding import seed_everything
from simworld.types import validate_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    ingest(cfg_obj)
    build_views(cfg_obj)


if __name__ == "__main__":
    main()
