"""Driver behavior (§15): skip/manifest, hard deps, caching, force_stage, isolated envs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from simworld import stages as stage_impls
from simworld.pipeline import run_pipeline
from simworld.tracking import NullTracker, Tracker
from simworld.types import SimWorldConfig, StagesCfg, validate_config


def test_all_disabled_writes_manifest(smoke_cfg: SimWorldConfig) -> None:
    manifest = run_pipeline(smoke_cfg.model_copy(update={"stages": StagesCfg()}), NullTracker())
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    assert all(r["status"] == "SKIPPED" for r in stages.values())
    assert (Path(smoke_cfg.paths.reports) / "run_manifest.json").exists()


def test_recon_stage_runs_and_unbuilt_stages_block(smoke_cfg: SimWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(
        update={"stages": StagesCfg(recon=True, emulator=True, rl=True, figures=True)}
    )
    manifest = run_pipeline(cfg, NullTracker())
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    assert stages["recon"]["status"] == "DONE"
    recon_out = json.loads(Path(stages["recon"]["outputs"][0]).read_text())
    assert "versions" in recon_out
    # figures (Stage 15) has no hard upstream deps and degrades gracefully on missing
    # artifacts, so a recon-only run still completes it (writing whatever it can).
    assert stages["figures"]["status"] == "DONE"
    # emulator is built now, but a recon-only run never produced its observed-world
    # inputs, so it fails fast; its enabled hard dependent blocks on the failure.
    assert stages["emulator"]["status"] == "FAILED"
    assert stages["rl"]["status"] == "BLOCKED"
    assert "hard dependency" in stages["rl"]["notes"]


# ---------------------------------------------------------------------------
# §15 caching contract, exercised against a tiny fake stage registry.


def _fake_stage(counter: dict[str, int], out: Path) -> Any:
    def run(cfg: SimWorldConfig, tracker: Tracker) -> list[Path]:
        counter["n"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("x")
        return [out]

    return run


def test_rerun_with_unchanged_config_is_cached(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    out = Path(smoke_cfg.paths.root) / "data" / "fake.parquet"
    monkeypatch.setattr(stage_impls, "stage_data", _fake_stage(calls, out))
    cfg = smoke_cfg.model_copy(update={"stages": StagesCfg(data=True)})

    first = run_pipeline(cfg, NullTracker())["stages"]
    second = run_pipeline(cfg, NullTracker())["stages"]
    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["data"]["status"] == "DONE"
    assert second["data"]["status"] == "CACHED"
    assert second["data"]["outputs"] == [str(out)]
    assert calls["n"] == 1  # the implementation ran exactly once


def test_watched_config_change_invalidates_cache(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    out = Path(smoke_cfg.paths.root) / "data" / "fake.parquet"
    monkeypatch.setattr(stage_impls, "stage_data", _fake_stage(calls, out))
    cfg = smoke_cfg.model_copy(update={"stages": StagesCfg(data=True)})

    run_pipeline(cfg, NullTracker())
    # `seed` is in the data stage's watched sections (STAGE_ORDER): the hash changes.
    changed = cfg.model_copy(update={"seed": cfg.seed + 1})
    rerun = run_pipeline(changed, NullTracker())["stages"]
    assert isinstance(rerun, dict)
    assert rerun["data"]["status"] == "DONE"
    assert calls["n"] == 2


def test_force_stage_reruns_stage_and_downstream(
    smoke_cfg: SimWorldConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    counters = {name: {"n": 0} for name in ("data", "graphs", "abm")}
    for name in counters:
        out = Path(smoke_cfg.paths.root) / name / "fake.parquet"
        monkeypatch.setattr(stage_impls, f"stage_{name}", _fake_stage(counters[name], out))
    cfg = smoke_cfg.model_copy(update={"stages": StagesCfg(data=True, graphs=True, abm=True)})

    run_pipeline(cfg, NullTracker())
    forced = cfg.model_copy(update={"force_stage": "graphs"})
    rerun = run_pipeline(forced, NullTracker())["stages"]
    assert isinstance(rerun, dict)
    assert rerun["data"]["status"] == "CACHED"  # upstream stays cached
    assert rerun["graphs"]["status"] == "DONE"  # the forced stage re-runs
    assert rerun["abm"]["status"] == "DONE"  # ... and everything downstream of it
    assert {n: c["n"] for n, c in counters.items()} == {"data": 1, "graphs": 2, "abm": 2}


def test_force_stage_unknown_name_raises(smoke_cfg: SimWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(update={"force_stage": "emulatr", "stages": StagesCfg()})
    with pytest.raises(ValueError, match="emulatr"):
        run_pipeline(cfg, NullTracker())


# ---------------------------------------------------------------------------
# isolated_envs (§5 fallback): per-group uv venvs, without ever creating one.


@pytest.fixture()
def subprocess_recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, str]]]:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(cmd), dict(kwargs.get("env") or {})))
        # stdout="" keeps the manifest's _git_head() call working under the patch
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(stage_impls, "_SYNCED_GROUPS", set())
    return calls


def test_isolated_stage_syncs_group_venv_and_runs_via_uv(
    smoke_cfg: SimWorldConfig,
    subprocess_recorder: list[tuple[list[str], dict[str, str]]],
) -> None:
    cfg = smoke_cfg.model_copy(update={"isolated_envs": True})
    stage_impls.isolated_stage("calibration")(cfg, NullTracker())

    sync_cmd, sync_env = subprocess_recorder[0]
    assert sync_cmd == ["uv", "sync", "--extra", "dev", "--extra", "bayes", "-q"]
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == ".venv-bayes"
    run_cmd, run_env = subprocess_recorder[1]
    assert run_cmd == ["uv", "run", "--no-sync", "python", "scripts/calibrate.py", "profile=smoke"]
    assert run_env["UV_PROJECT_ENVIRONMENT"] == ".venv-bayes"
    assert len(subprocess_recorder) == 2


def test_isolated_stage_core_group_and_multi_script(
    smoke_cfg: SimWorldConfig,
    subprocess_recorder: list[tuple[list[str], dict[str, str]]],
) -> None:
    cfg = smoke_cfg.model_copy(update={"isolated_envs": True})
    stage_impls.isolated_stage("data")(cfg, NullTracker())  # no extras group -> "core"

    cmds = [cmd for cmd, _ in subprocess_recorder]
    assert cmds[0] == ["uv", "sync", "--extra", "dev", "-q"]  # no --extra <group> for core
    assert [c[4] for c in cmds[1:]] == ["scripts/generate_world.py", "scripts/make_data.py"]
    assert all(env["UV_PROJECT_ENVIRONMENT"] == ".venv-core" for _, env in subprocess_recorder)


def test_isolated_sync_issued_once_per_group(
    smoke_cfg: SimWorldConfig,
    subprocess_recorder: list[tuple[list[str], dict[str, str]]],
) -> None:
    cfg = smoke_cfg.model_copy(update={"isolated_envs": True})
    stage_impls.isolated_stage("rl")(cfg, NullTracker())  # group "rl"
    stage_impls.isolated_stage("ensemble")(cfg, NullTracker())  # group "rl" again
    stage_impls.isolated_stage("sensitivity")(cfg, NullTracker())  # group "opt"

    syncs = [cmd for cmd, _ in subprocess_recorder if cmd[:2] == ["uv", "sync"]]
    assert syncs == [
        ["uv", "sync", "--extra", "dev", "--extra", "rl", "-q"],
        ["uv", "sync", "--extra", "dev", "--extra", "opt", "-q"],
    ]


def test_c6_marl_ablation_wired_into_rl_stage() -> None:
    """Regression guard: the Stage-10d MARL ablation must stay wired into the driver.

    It was once orphaned — only ``scripts/train_marl.py`` existed, never a stage —
    so ``artifacts/marl/c6_comparison.json`` was never produced and claim C6 was
    unanswerable at every scale. The rl stage carries ``train_marl.py`` in its
    script list (isolated path) and runs it in the ``rl`` extras group (SB3).
    """
    scripts, group = stage_impls.STAGE_SCRIPTS["rl"]
    assert "train_rl.py" in scripts
    assert "train_marl.py" in scripts, "C6 MARL ablation orphaned again — see stage_rl"
    assert group == "rl"


def test_c1_recovery_grid_flag_off_at_smoke_on_at_dev() -> None:
    """The C1 contrast is gated: off at smoke (< 6 min budget), on at dev."""
    from tests.conftest import compose_cfg

    smoke = validate_config(compose_cfg("profile=smoke"))
    dev = validate_config(compose_cfg("profile=dev"))
    assert smoke.calibration.recovery_grid is False
    assert dev.calibration.recovery_grid is True


def test_c1_recovery_grid_wired_into_calibration_stage() -> None:
    """Regression guard: stage_calibration must run the recovery grid when enabled.

    C1 is a two-world contrast; a single pipeline run ships one dgp variant, so
    without this the wellspecified-vs-confounded contrast is un-producible at any scale.
    """
    import inspect

    src = inspect.getsource(stage_impls.stage_calibration)
    assert "recovery_grid.py" in src, "C1 recovery grid orphaned from stage_calibration"
    assert "cfg.calibration.recovery_grid" in src


def test_c1_recovery_grid_contrast_is_consumed(smoke_cfg: SimWorldConfig) -> None:
    """evaluate() reports the two-world contrast when recovery_grid.json is present:
    coverage + convergence from the wellspecified cell, β_peer miss from the confounded
    cell — not the shipped single variant."""
    from simworld.evaluation import parameter_recovery

    calib = Path(smoke_cfg.paths.root) / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    grid = {
        "schema": parameter_recovery.GRID_SCHEMA,
        "cells": {
            "wellspecified": {
                "coverage_at_90": "15/17",
                "coverage_fraction": 0.882,
                "max_r_hat": 1.005,
                "divergences": 0,
                "per_parameter": [{"parameter": "beta_peer", "hdi_90_covers": True}],
                "beta_peer_covers": True,
                "beta_peer_bias": 0.01,
            },
            "confounded": {
                "coverage_at_90": "13/17",
                "beta_peer_covers": False,
                "beta_peer_bias": -0.63,
            },
        },
        "contrast": {"clean_contrast": True},
    }
    (calib / "recovery_grid.json").write_text(json.dumps(grid))
    result = parameter_recovery.evaluate(smoke_cfg)
    assert isinstance(result["mode"], str) and result["mode"].startswith("contrast")
    assert result["coverage_at_90"] == "15/17"  # wellspecified recovery half
    assert result["max_r_hat"] == 1.005  # convergence judged on the wellspecified fit
    assert result["beta_peer_miss_under_confounded"] is True  # confounded failure half


def test_pipeline_routes_script_stages_through_uv_when_isolated(
    smoke_cfg: SimWorldConfig,
    subprocess_recorder: list[tuple[list[str], dict[str, str]]],
) -> None:
    cfg = smoke_cfg.model_copy(
        update={"isolated_envs": True, "stages": StagesCfg(figures=True, report=True)}
    )
    manifest = run_pipeline(cfg, NullTracker())["stages"]
    assert isinstance(manifest, dict)
    assert manifest["figures"]["status"] == "DONE"
    assert manifest["report"]["status"] == "DONE"
    scripts = [cmd[4] for cmd, _ in subprocess_recorder if cmd[:3] == ["uv", "run", "--no-sync"]]
    assert scripts == ["scripts/make_figures.py", "scripts/build_report.py"]


def test_non_isolated_run_script_uses_current_interpreter(
    smoke_cfg: SimWorldConfig,
    subprocess_recorder: list[tuple[list[str], dict[str, str]]],
) -> None:
    assert smoke_cfg.isolated_envs is False
    stage_impls._run_script(smoke_cfg, "calibrate.py", group="bayes")
    assert len(subprocess_recorder) == 1  # no uv sync, no uv run
    cmd, env = subprocess_recorder[0]
    assert cmd == [sys.executable, "scripts/calibrate.py", "profile=smoke"]
    assert env.get("JAX_PLATFORMS") == "cpu"  # §5 JAX isolation is unchanged
