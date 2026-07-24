"""§10 Stage 15 driver: build every available Matplotlib + Plotly figure.

Writes ``reports/figures/*.png`` (the 13 paper figures) and
``reports/figures/*.html`` (Plotly exploration). A missing upstream artifact
degrades that one figure — logged and skipped — rather than crashing the run.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from simworld.logging_conf import get_logger, setup_logging
from simworld.seeding import seed_everything
from simworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()

    from simworld.visualization.figures import make_all_figures
    from simworld.visualization.interactive import make_all_interactive

    static_paths = make_all_figures(cfg_obj)
    log.info("wrote %d/13 static figures", len(static_paths))
    for path in static_paths:
        log.info("figure -> %s", path)

    interactive_paths = make_all_interactive(cfg_obj)
    log.info("wrote %d interactive figures", len(interactive_paths))
    for path in interactive_paths:
        log.info("interactive -> %s", path)


if __name__ == "__main__":
    main()
