"""§14 Stage 14 driver: SALib Morris → Sobol sensitivity + Optuna policy search.

Hydra entry point. Runs Morris screening and Sobol analysis on the trained emulator,
validates with ABM cross-check, and optionally runs Optuna policy search.
"""

from __future__ import annotations

import json
import sys

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

        # MERGE the Optuna results into the module's sensitivity_summary.json
        # rather than clobbering it with a different schema — report.py reads
        # both the Morris/Sobol analysis fields and optuna_best_J from this one
        # canonical file.
        summary_path = sensitivity_result.summary
        summary = json.loads(summary_path.read_text())
        summary.update(
            {
                "indices_path": str(sensitivity_result.indices),
                "optuna_path": str(optuna_path),
                "optuna_best_J": optuna_result["best_J"],
                "optuna_best_levers": optuna_result["best_levers"],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Stage-14 summary (analysis + Optuna) -> %s", summary_path)

        log.info("§14 Stage 14 complete: exit 0")
        sys.exit(0)

    except Exception as e:
        log.exception("§14 Stage 14 FAILED: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
