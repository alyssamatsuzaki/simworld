"""C1 recovery grid: calibrate under BOTH the wellspecified and confounded worlds.

The C1 claim is a *contrast* — Bayesian calibration recovers theta* when the model is
well specified, and fails legibly (a biased peer coefficient beta_peer) when supply-
network capacity homophily is switched on. A single pipeline run ships ONE dgp variant,
so this contrast is not producible from the main artifacts alone. This entry point
re-runs the world -> data -> calibrate chain under each variant into a dedicated
sub-root, then hands the posteriors to ``evaluation.parameter_recovery.build_grid`` for
grading against the sealed theta* (identical across variants — the behavioral truth is
shared; only the world's structure differs).

Gated by ``calibration.recovery_grid`` (off at smoke to protect the < 6 min budget; on
at dev). Grading reads the answer key, so it lives in ``evaluation/`` behind the
firewall — this script only orchestrates subprocesses and never touches the key itself.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from simworld.evaluation.parameter_recovery import GRID_CELLS, build_grid
from simworld.logging_conf import setup_logging
from simworld.seeding import seed_everything
from simworld.types import SimWorldConfig, validate_config

log = logging.getLogger(__name__)

# world -> data build the observed inputs; calibrate fits the posterior. Crosscheck is
# off (the PyMC re-fit is not needed for coverage grading) to keep the grid affordable.
_CELL_SCRIPTS = ("generate_world.py", "make_data.py", "calibrate.py")


def _cell_root(cfg: SimWorldConfig, variant: str) -> str:
    return str(Path(cfg.paths.root) / "recovery_grid" / variant)


def _build_cell(cfg: SimWorldConfig, variant: str, root: str) -> None:
    """Run the world+data+calibrate chain for one variant into its own root."""
    for script in _CELL_SCRIPTS:
        cmd = [
            sys.executable,
            f"scripts/{script}",
            f"profile={cfg.profile_name}",
            f"dgp={variant}",
            f"paths.root={root}",
            f"seed={cfg.seed}",
            "calibration.crosscheck=false",
        ]
        log.info("recovery-grid[%s]: %s", variant, " ".join(cmd))
        subprocess.run(cmd, check=True)


def _run(cfg: SimWorldConfig) -> Path:
    cell_roots: dict[str, str] = {}
    for variant in GRID_CELLS:
        root = _cell_root(cfg, variant)
        _build_cell(cfg, variant, root)
        cell_roots[variant] = root
    return build_grid(cfg, cell_roots)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    out = _run(cfg_obj)
    log.info("recovery grid written: %s", out)


if __name__ == "__main__":
    main()
