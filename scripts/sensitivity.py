"""§14 Stage 14 driver: SALib Morris → Sobol sensitivity + Optuna policy search.

Hydra entry point. Runs Morris screening and Sobol analysis on the trained emulator,
validates with ABM cross-check, and optionally runs Optuna policy search.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.sensitivity.policy_search import run_policy_search, save_optuna_best
from regworld.sensitivity.screen import run_sensitivity
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()

    try:
        sensitivity_result = run_sensitivity(cfg_obj)
        log.info("Sensitivity screening completed")

        optuna_result = run_policy_search(cfg_obj)
        optuna_path = save_optuna_best(cfg_obj, optuna_result)
        log.info("Optuna policy search completed")

        summary = {
            "profile": cfg_obj.profile_name,
            "seed": cfg_obj.seed,
            "indices_path": str(sensitivity_result.indices),
            "summary_path": str(sensitivity_result.summary),
            "optuna_path": str(optuna_path),
            "metrics": sensitivity_result.metrics,
            "optuna_best_J": optuna_result["best_J"],
            "optuna_best_levers": optuna_result["best_levers"],
        }

        summary_path = Path(cfg_obj.paths.root) / "sensitivity" / "sensitivity_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Full summary → %s", summary_path)

        log.info("§14 Stage 14 complete: exit 0")
        sys.exit(0)

    except Exception as e:
        log.exception("§14 Stage 14 FAILED: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
