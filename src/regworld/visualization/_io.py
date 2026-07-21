"""Small shared IO helpers for the visualization package (Stage 15).

Every loader here degrades honestly: a missing or unreadable artifact returns
``None`` (or an empty frame) and logs a warning instead of raising, so a
single absent input never takes down the whole figure/report run. Nothing in
this module imports ``regworld.dgp`` or reads the sealed answer-key tree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact; ``None`` (with a warning) if missing or malformed."""
    if not path.is_file():
        log.warning("artifact missing, skipping: %s", path)
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("artifact unreadable, skipping %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        log.warning("artifact %s did not decode to a mapping, skipping", path)
        return None
    return payload


def load_cube(cfg: RegWorldConfig) -> Any | None:
    """Load the Stage-11 scenario cube (``artifacts/ensemble/cube.parquet``)."""
    import polars as pl

    path = Path(cfg.paths.root) / "ensemble" / "cube.parquet"
    if not path.is_file():
        log.warning("scenario cube missing, skipping: %s", path)
        return None
    try:
        return pl.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive against a corrupt file
        log.warning("scenario cube unreadable, skipping %s: %s", path, exc)
        return None


def action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """The 4-lever action box (enforcement, targeting, phase_speed, subsidy)."""
    from regworld.training.datamodule import ACTION_HIGH, ACTION_LOW

    return np.asarray(ACTION_LOW, dtype=np.float64), np.asarray(ACTION_HIGH, dtype=np.float64)


def eval_metrics(cfg: RegWorldConfig) -> dict[str, Any] | None:
    """The §11 evaluation report (``reports/eval/metrics.json``)."""
    return load_json(Path(cfg.paths.reports) / "eval" / "metrics.json")
