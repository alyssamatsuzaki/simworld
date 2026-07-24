"""§17 driver: assemble FINDINGS.md from committed artifacts.

Mirrors the eval_emulator.py structure: Hydra config, validate, seed, setup logging,
then call build_findings(cfg_obj).
"""

from __future__ import annotations

import sys

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

    from simworld.evaluation.report import build_findings

    try:
        findings_path = build_findings(cfg_obj)
        log.info("build_findings -> %s", findings_path)
        sys.exit(0)
    except Exception as e:
        log.exception("build_findings FAILED: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
