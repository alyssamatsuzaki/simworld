"""§10 Stage 11 driver: build the scenario cube and its ABM cross-validation.

Writes ``artifacts/ensemble/{cube.parquet, validation_report.json,
ensemble_summary.json}``. A missing emulator checkpoint is an honest partial
(``simworld.pipeline.Degraded``), logged and reported with a nonzero exit
rather than silently skipped.
"""

from __future__ import annotations

import json
import sys

import hydra
from omegaconf import DictConfig

from simworld.logging_conf import get_logger, setup_logging
from simworld.pipeline import Degraded
from simworld.seeding import seed_everything
from simworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()

    from simworld.ensemble import run_ensemble
    from simworld.ensemble.validation import CoverageGateFailure, enforce_coverage_gate

    try:
        result = run_ensemble(cfg_obj)
    except Degraded as exc:
        log.warning("ensemble DEGRADED: %s", exc)
        sys.exit(1)

    log.info("ensemble cube -> %s", result.cube)
    log.info("ensemble summary -> %s", result.summary)
    log.info("ensemble metrics: %s", json.dumps(result.metrics, indent=2))

    # Enforce the >=0.85 coverage gate here too, not only inside the pipeline
    # driver — a standalone `make ensemble` at a gating profile must fail loudly
    # when the emulator's predictive band does not cover the ABM (§10 Stage 11).
    try:
        status = enforce_coverage_gate(cfg_obj, result.metrics["coverage"])
        log.info("coverage gate: %s (coverage=%.3f)", status, result.metrics["coverage"])
    except CoverageGateFailure as exc:
        log.error("coverage gate FAILED: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
