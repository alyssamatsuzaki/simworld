"""Stage 5f: the four-number simulator gate. FLAGS write reports/simulator_discrepancy.md."""

import hydra
from omegaconf import DictConfig

from regworld.causal.gate import run_gate, write_gate_outputs
from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    result = run_gate(cfg_obj)
    write_gate_outputs(cfg_obj, result)
    log.info(
        "gate %s: tau_true=%.4f tau_abm=%.4f tau_qe=%.4f tau_obs=%.4f",
        "FLAGGED" if result.flagged else "PASSED",
        result.tau_true,
        result.tau_abm,
        result.tau_qe,
        result.tau_obs,
    )


if __name__ == "__main__":
    main()
