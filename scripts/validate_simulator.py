"""Stage 5f: the four-number simulator gate. FLAGS write reports/simulator_discrepancy.md."""

import hydra
from omegaconf import DictConfig

from simworld.causal.gate import run_gate, write_gate_outputs
from simworld.logging_conf import get_logger, setup_logging
from simworld.seeding import seed_everything
from simworld.types import validate_config

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
    # PLAN 5f names two disagreement policies. `report` (continue with the
    # discrepancy artifact + downstream warning banners) is fully wired via
    # write_gate_outputs. `recalibrate` (fold the DiD in as a Stage-4b
    # moment-matching penalty and re-run 4->5 once) is NOT automated in this
    # build; rather than loop or silently no-op, we log loudly and continue in
    # report mode so the FLAG is never hidden. See docs/DEVIATIONS.md.
    if result.flagged and cfg_obj.causal.on_disagreement == "recalibrate":
        log.warning(
            "on_disagreement=recalibrate is not automated in this build; the run "
            "is FLAGGED and continues in report mode. Re-run manually with the DiD "
            "moment penalty (calibration.did_penalty>0) to recalibrate, per PLAN 5f."
        )


if __name__ == "__main__":
    main()
