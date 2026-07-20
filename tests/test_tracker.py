"""Null tracker writes nothing; MLflow file backend writes a run dir (§13)."""

from __future__ import annotations

from pathlib import Path

from regworld.tracking import MlflowTracker, NullTracker


def test_null_tracker_writes_nothing(tmp_path: Path) -> None:
    t = NullTracker()
    t.start("run", {"a": 1})
    t.log_metrics({"x": 1.0})
    t.finish()
    assert list(tmp_path.iterdir()) == []


def test_mlflow_tracker_writes_run(tmp_path: Path) -> None:
    # sqlite backend: the file store is in maintenance mode in mlflow 3.x (see DEVIATIONS.md)
    db = tmp_path / "mlflow.db"
    t = MlflowTracker(f"sqlite:///{db}", "regworld-test")
    t.start("unit", {"nested": {"k": 1}, "seed": 0})
    t.log_metrics({"loss": 0.5}, step=1)
    t.log_metrics({"bad": float("nan")})  # silently dropped, never crashes
    t.finish()
    assert db.exists()
    import mlflow

    runs = mlflow.search_runs(experiment_names=["regworld-test"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.loss"] == 0.5
