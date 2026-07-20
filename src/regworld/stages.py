"""Stage implementations for the driver. Filled in phase by phase (§10).

Contract: `stage_<name>(cfg, tracker) -> list[Path]` of durable outputs.
Raise `regworld.pipeline.Degraded(note)` for an honest partial result.
Heavy imports stay inside the functions: the driver process must not pay for
(or conflict with) libraries a disabled stage would have used. Calibration is
always launched as a subprocess so JAX never enters this process (§5).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from regworld.tracking import Tracker
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def _run_script(
    cfg: RegWorldConfig,
    script: str,
    extra_overrides: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a stage script as a subprocess with the current profile (JAX isolation, §5)."""
    cmd = [sys.executable, f"scripts/{script}", f"profile={cfg.profile_name}"]
    cmd += extra_overrides or []
    full_env = dict(os.environ)
    full_env.setdefault("JAX_PLATFORMS", "cpu")
    full_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    full_env.update(env or {})
    log.info("subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=full_env)


def stage_recon(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    """Record the installed stack and any skipped extras into the run."""
    skips = Path(".stage_skips")
    skipped = skips.read_text().split() if skips.exists() else []
    out = Path(cfg.paths.root) / "recon.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "polars", "networkx", "mesa", "torch", "gymnasium", "mlflow"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as e:
            versions[mod] = f"unavailable: {type(e).__name__}"
    out.write_text(json.dumps({"skipped_extras": skipped, "versions": versions}, indent=2))
    return [out]


def stage_data(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 2, Stage 1")


def stage_graphs(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 2, Stage 2")


def stage_abm(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3, Stage 3")


def stage_tensorized_abm(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3, Stage 3b")


def stage_calibration(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 4, Stage 4")


def stage_causal(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 4, Stage 5")


def stage_emulator(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 5, Stages 6-7")


def stage_envs(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3/5, Stage 8")


def stage_marl(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 3, Stage 9")


def stage_rl(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 10")


def stage_ensemble(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 11")


def stage_sensitivity(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 6, Stage 14")


def stage_figures(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 7, Stage 15")


def stage_report(cfg: RegWorldConfig, tracker: Tracker) -> list[Path]:
    raise NotImplementedError("Phase 7, Stage 17")
