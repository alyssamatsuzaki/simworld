"""Stage 1b: observed tables → analysis-ready panel + DuckDB views."""

import hydra
from omegaconf import DictConfig

from regworld.data.duck import build_views
from regworld.data.ingest import ingest
from regworld.logging_conf import setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    ingest(cfg_obj)
    build_views(cfg_obj)


if __name__ == "__main__":
    main()
