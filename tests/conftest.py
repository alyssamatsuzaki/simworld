"""Shared fixtures: Hydra composition helper + a tmp-rooted validated config."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from regworld.types import RegWorldConfig, validate_config

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def compose_cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@pytest.fixture()
def smoke_cfg(tmp_path: Path) -> RegWorldConfig:
    """Validated smoke-profile config with artifact/report roots inside tmp_path."""
    cfg = validate_config(compose_cfg("profile=smoke", "tracking=none"))
    cfg.paths.root = str(tmp_path / "artifacts")
    cfg.paths.data = str(tmp_path / "artifacts/data")
    cfg.paths.graphs = str(tmp_path / "artifacts/graphs")
    cfg.paths.reports = str(tmp_path / "reports")
    return cfg
