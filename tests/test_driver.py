"""Driver behavior (§15): empty stages runs nothing; recon runs; manifest is written."""

from __future__ import annotations

import json
from pathlib import Path

from regworld.pipeline import run_pipeline
from regworld.tracking import NullTracker
from regworld.types import RegWorldConfig, StagesCfg


def test_all_disabled_writes_manifest(smoke_cfg: RegWorldConfig) -> None:
    manifest = run_pipeline(smoke_cfg.model_copy(update={"stages": StagesCfg()}), NullTracker())
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    assert all(r["status"] == "SKIPPED" for r in stages.values())
    assert (Path(smoke_cfg.paths.reports) / "run_manifest.json").exists()


def test_recon_stage_runs_and_unbuilt_stages_block(smoke_cfg: RegWorldConfig) -> None:
    cfg = smoke_cfg.model_copy(
        update={"stages": StagesCfg(recon=True, emulator=True, rl=True, figures=True)}
    )
    manifest = run_pipeline(cfg, NullTracker())
    stages = manifest["stages"]
    assert isinstance(stages, dict)
    assert stages["recon"]["status"] == "DONE"
    recon_out = json.loads(Path(stages["recon"]["outputs"][0]).read_text())
    assert "versions" in recon_out
    # figures (Stage 15) is not built yet and has no upstream, so it blocks honestly.
    assert stages["figures"]["status"] == "BLOCKED"
    assert "not built" in stages["figures"]["notes"]
    # emulator is built now, but a recon-only run never produced its observed-world
    # inputs, so it fails fast; its enabled hard dependent blocks on the failure.
    assert stages["emulator"]["status"] == "FAILED"
    assert stages["rl"]["status"] == "BLOCKED"
    assert "hard dependency" in stages["rl"]["notes"]
