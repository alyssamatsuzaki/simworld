"""Every config group value composes and validates; a bogus key dies immediately (§6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from regworld.types import validate_config

from .conftest import CONFIG_DIR, compose_cfg

GROUPS = sorted(
    p.name for p in Path(CONFIG_DIR).iterdir() if p.is_dir() and not p.name.startswith("_")
)


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (Path(CONFIG_DIR) / group).glob("*.yaml"))


@pytest.mark.parametrize(
    "group,option",
    [(g, o) for g in GROUPS for o in _options(g)],
)
def test_every_group_option_validates(group: str, option: str) -> None:
    cfg = validate_config(compose_cfg(f"{group}={option}"))
    assert cfg.horizon_quarters == 24


def test_default_composition_validates() -> None:
    cfg = validate_config(compose_cfg())
    assert cfg.profile_name == "smoke"
    assert cfg.dgp.variant == "confounded"
    assert cfg.network.homophily == 1.5  # dgp=confounded sets the homophily knob


def test_profile_wins_on_size_knobs() -> None:
    smoke = validate_config(compose_cfg("profile=smoke"))
    assert smoke.population.n_firms == 200  # profile overrides population/base's 2000


def test_wellspecified_disables_both_confounds() -> None:
    cfg = validate_config(compose_cfg("dgp=wellspecified"))
    assert cfg.network.homophily == 0.0
    assert cfg.dgp.corr_z_size == 0.0


def test_bogus_key_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config(compose_cfg("+emulator_latent_dimz=64"))


def test_bogus_nested_key_raises() -> None:
    with pytest.raises(ValidationError):
        validate_config(compose_cfg("+emulator.latent_dimz=64"))


def test_stages_delete_override_runs_nothing() -> None:
    # Hydra merges `stages={}` into the all-true defaults (a no-op), so the no-op
    # override is `~stages`: delete the node, fall back to StagesCfg all-False defaults.
    cfg = validate_config(compose_cfg("~stages"))
    assert not any(cfg.stages.model_dump().values())
