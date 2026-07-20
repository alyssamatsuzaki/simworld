"""One tracking interface (§13). No stage imports mlflow/wandb directly.

MLflow file backend needs no credentials and no network, which is what keeps the
one-command run alive on any cluster. W&B falls back to offline without an API key.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def _git_head() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
            or "unknown"
        )
    except OSError:  # pragma: no cover
        return "unknown"


class Tracker(Protocol):
    def start(self, run_name: str, config: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...

    def log_figure(self, fig: Any, name: str) -> None: ...

    def log_artifact(self, path: Path, name: str | None = None) -> None: ...

    def log_table(self, rows: list[dict[str, Any]], name: str) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Backend for tests: records nothing, accepts everything."""

    def start(self, run_name: str, config: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        pass

    def log_figure(self, fig: Any, name: str) -> None:
        pass

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        pass

    def log_table(self, rows: list[dict[str, Any]], name: str) -> None:
        pass

    def finish(self) -> None:
        pass


class MlflowTracker:
    def __init__(self, uri: str, experiment: str) -> None:
        import mlflow  # local import: mlflow costs seconds and only this backend pays it

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._active = False

    def start(self, run_name: str, config: dict[str, Any]) -> None:
        self._mlflow.start_run(run_name=run_name)
        self._active = True
        flat = _flatten(config)
        # mlflow caps params per batch; log in chunks
        items = list(flat.items())
        for i in range(0, len(items), 90):
            self._mlflow.log_params(dict(items[i : i + 90]))
        self._mlflow.set_tag("git_commit", _git_head())

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if _is_finite_number(v)}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_figure(self, fig: Any, name: str) -> None:
        self._mlflow.log_figure(fig, name if name.endswith(".png") else f"{name}.png")

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        if Path(path).is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=name)
        else:
            self._mlflow.log_artifact(str(path), artifact_path=name)

    def log_table(self, rows: list[dict[str, Any]], name: str) -> None:
        self._mlflow.log_table(
            data={k: [r.get(k) for r in rows] for k in rows[0]} if rows else {},
            artifact_file=f"{name}.json",
        )

    def finish(self) -> None:
        if self._active:
            self._mlflow.end_run()
            self._active = False


class WandbTracker:
    def __init__(self, experiment: str) -> None:
        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        self._wandb = wandb
        self._experiment = experiment
        self._run: Any = None

    def start(self, run_name: str, config: dict[str, Any]) -> None:
        self._run = self._wandb.init(project=self._experiment, name=run_name, config=config)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._wandb.log(dict(metrics), step=step)

    def log_figure(self, fig: Any, name: str) -> None:
        self._wandb.log({name: self._wandb.Image(fig)})

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        art = self._wandb.Artifact(name or Path(path).stem, type="artifact")
        if Path(path).is_dir():
            art.add_dir(str(path))
        else:
            art.add_file(str(path))
        self._wandb.log_artifact(art)

    def log_table(self, rows: list[dict[str, Any]], name: str) -> None:
        if rows:
            table = self._wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])
            self._wandb.log({name: table})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def make_tracker(cfg: RegWorldConfig) -> Tracker:
    backend = cfg.tracking.backend
    if backend == "none":
        return NullTracker()
    if backend == "wandb":
        try:
            return WandbTracker(cfg.tracking.experiment)
        except ImportError:
            log.warning("wandb not installed; falling back to the null tracker")
            return NullTracker()
    return MlflowTracker(cfg.tracking.uri, cfg.tracking.experiment)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten(v, key))
        else:
            out[key] = str(v)[:250]
    return out


def _is_finite_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))
