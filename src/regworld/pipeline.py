"""The driver (§15): stage order, hard dependencies, checkpoint/skip logic, run manifest.

Stage implementations live in `regworld.stages`; each returns a list of output paths.
A stage that is not yet built raises NotImplementedError and is recorded BLOCKED —
the driver never fakes DONE.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from regworld import stages as stage_impls
from regworld.tracking import Tracker, _git_head
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

# Build order. Each entry: (stage key in cfg.stages, config sections whose change invalidates it).
STAGE_ORDER: list[tuple[str, list[str]]] = [
    ("recon", []),
    (
        "data",
        ["seed", "horizon_quarters", "observed_quarters", "data", "dgp", "population", "network"],
    ),
    ("graphs", ["seed", "dgp", "population", "network"]),
    (
        "abm",
        [
            "seed",
            "horizon_quarters",
            "dgp",
            "population",
            "network",
            "behavior",
            "abm",
            "objective",
            "policy",
        ],
    ),
    ("tensorized_abm", ["seed", "horizon_quarters", "dgp", "population", "network", "behavior"]),
    (
        "calibration",
        ["seed", "observed_quarters", "dgp", "population", "network", "behavior", "calibration"],
    ),
    ("causal", ["seed", "dgp", "population", "network", "causal"]),
    (
        "emulator",
        ["seed", "horizon_quarters", "dgp", "population", "network", "behavior", "emulator"],
    ),
    ("envs", ["seed", "env", "objective"]),
    ("marl", ["seed", "env", "rl", "objective"]),
    ("rl", ["seed", "rl", "objective", "emulator"]),
    ("ensemble", ["seed", "seeds", "ensemble", "objective", "emulator"]),
    ("sensitivity", ["seed", "sensitivity", "emulator"]),
    ("figures", []),  # always cheap; re-run every time
    ("report", []),
]

# stage -> stages it hard-depends on (§15): a FAILED/BLOCKED dependency forces BLOCKED.
HARD_DEPS: dict[str, list[str]] = {
    "graphs": ["data"],
    "abm": ["data", "graphs"],
    "tensorized_abm": ["data", "graphs", "abm"],
    "calibration": ["data", "graphs", "abm", "tensorized_abm"],
    "causal": ["data", "calibration"],
    "emulator": ["abm", "calibration"],
    "envs": ["abm"],
    "marl": ["abm"],
    "rl": ["emulator"],
    "ensemble": ["emulator"],
    "sensitivity": ["emulator"],
}

_NEVER_CACHE = {"recon", "figures", "report"}


@dataclass
class StageResult:
    name: str
    status: str  # DONE | SKIPPED | CACHED | DEGRADED | FAILED | BLOCKED
    wall_clock: float = 0.0
    outputs: list[str] = field(default_factory=list)
    notes: str = ""


class Degraded(Exception):
    """Raise from a stage to finish with status DEGRADED; args[0] is the note."""

    def __init__(self, note: str, outputs: list[str] | None = None) -> None:
        super().__init__(note)
        self.outputs = outputs or []


def _state_dir(cfg: RegWorldConfig) -> Path:
    d = Path(cfg.paths.root) / ".stage_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cfg_hash(cfg: RegWorldConfig, sections: list[str]) -> str:
    def dump(s: str) -> object:
        v = getattr(cfg, s)
        return v.model_dump() if hasattr(v, "model_dump") else v

    payload: dict[str, object] = {s: dump(s) for s in sections}
    payload["profile_name"] = cfg.profile_name
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cached(cfg: RegWorldConfig, name: str, sections: list[str]) -> list[str] | None:
    """Outputs of a previous identical run, or None if the stage must run."""
    if name in _NEVER_CACHE:
        return None
    f = _state_dir(cfg) / f"{name}.json"
    if not f.exists():
        return None
    try:
        state = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None
    if state.get("hash") != _cfg_hash(cfg, sections):
        return None
    outputs = [str(p) for p in state.get("outputs", [])]
    if outputs and all(Path(p).exists() for p in outputs):
        return outputs
    return None


def _save_state(cfg: RegWorldConfig, name: str, sections: list[str], outputs: list[str]) -> None:
    f = _state_dir(cfg) / f"{name}.json"
    f.write_text(json.dumps({"hash": _cfg_hash(cfg, sections), "outputs": outputs}, indent=2))


def run_pipeline(cfg: RegWorldConfig, tracker: Tracker) -> dict[str, object]:
    """Run every enabled stage in order; never let one broken stage kill the run silently."""
    results: dict[str, StageResult] = {}
    forced = False
    t_start = time.time()

    for name, sections in STAGE_ORDER:
        if cfg.force_stage == name:
            forced = True  # this stage and everything downstream re-runs
        if not getattr(cfg.stages, name):
            results[name] = StageResult(name, "SKIPPED", notes="disabled in config")
            continue

        bad_deps = [
            d
            for d in HARD_DEPS.get(name, [])
            if getattr(cfg.stages, d)
            and results.get(d) is not None
            and results[d].status in ("FAILED", "BLOCKED")
        ]
        if bad_deps:
            results[name] = StageResult(
                name, "BLOCKED", notes=f"hard dependency failed: {bad_deps}"
            )
            log.error("stage %-14s BLOCKED (deps: %s)", name, bad_deps)
            continue

        if not forced:
            cached_outputs = _cached(cfg, name, sections)
            if cached_outputs is not None:
                results[name] = StageResult(name, "CACHED", outputs=cached_outputs)
                log.info("stage %-14s CACHED", name)
                continue

        fn = getattr(stage_impls, f"stage_{name}", None)
        t0 = time.time()
        if fn is None:
            results[name] = StageResult(name, "BLOCKED", notes="no implementation registered")
            continue
        try:
            outputs = [str(p) for p in (fn(cfg, tracker) or [])]
            results[name] = StageResult(name, "DONE", time.time() - t0, outputs)
            _save_state(cfg, name, sections, outputs)
            log.info("stage %-14s DONE in %.1fs", name, time.time() - t0)
        except Degraded as e:
            outs = [str(p) for p in e.outputs]
            results[name] = StageResult(name, "DEGRADED", time.time() - t0, outs, str(e))
            _save_state(cfg, name, sections, outs)
            log.warning("stage %-14s DEGRADED: %s", name, e)
        except NotImplementedError as e:
            results[name] = StageResult(name, "BLOCKED", time.time() - t0, notes=f"not built: {e}")
            log.warning("stage %-14s BLOCKED (not built yet)", name)
        except Exception as e:
            results[name] = StageResult(
                name, "FAILED", time.time() - t0, notes=f"{type(e).__name__}: {e}"
            )
            log.exception("stage %-14s FAILED", name)

    manifest: dict[str, object] = {
        "profile": cfg.profile_name,
        "seed": cfg.seed,
        "git_commit": _git_head(),
        "wall_clock_total": round(time.time() - t_start, 1),
        "stages": {n: asdict(r) for n, r in results.items()},
    }
    reports = Path(cfg.paths.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("── run manifest ──")
    for n, r in results.items():
        log.info("  %-14s %-8s %6.1fs  %s", n, r.status, r.wall_clock, r.notes)
    return manifest
