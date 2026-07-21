"""§11 family 12 — sensitivity analysis, quick in-process version.

If artifacts/sensitivity/indices.json exists, read and report it.
Otherwise, run a small Sobol analysis (N=64 for smoke, ~1000 for dev) on the emulator
to provide real numbers even when `make sensitivity` hasn't run yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

from regworld.environments.emulator_env import EmulatorEnv
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.training.datamodule import ACTION_HIGH, ACTION_LOW
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def evaluate(cfg: RegWorldConfig) -> dict[str, object]:
    """Sensitivity evaluation: read artifacts or compute a quick Sobol analysis."""
    artifacts_path = Path(cfg.paths.root) / "sensitivity" / "indices.json"

    if artifacts_path.exists():
        log.info("Reading sensitivity indices from artifacts")
        payload = json.loads(artifacts_path.read_text())
        sobol = payload.get("sobol", {})
        S1 = sobol.get("S1", {})
        ST = sobol.get("ST", {})
        top_driver = max(S1.items(), key=lambda x: x[1])[0] if S1 else "unknown"
        abm_check = payload.get("abm_check", {})

        return {
            "status": "read from artifacts/sensitivity/indices.json",
            "method": "Morris → Sobol (full run)",
            "S1": S1,
            "ST": ST,
            "top_driver_S1": top_driver,
            "abm_check_spearman_corr": abm_check.get("emulator_vs_abm_spearman_corr"),
            "thresholds_dev": {
                "S1_positive": "all indices ∈ [0, 1]",
                "ST_gte_S1": "ST ≥ S1 within MC error",
                "abm_check_corr": "> 0.7 expected",
            },
        }

    log.info("Sensitivity artifacts not found; running quick Sobol (N=64)")
    problem = {
        "num_vars": 4,
        "names": ["enforcement", "targeting", "phase_speed", "subsidy"],
        "bounds": [
            [float(ACTION_LOW[0]), float(ACTION_HIGH[0])],
            [float(ACTION_LOW[1]), float(ACTION_HIGH[1])],
            [float(ACTION_LOW[2]), float(ACTION_HIGH[2])],
            [float(ACTION_LOW[3]), float(ACTION_HIGH[3])],
        ],
    }

    samples = sobol_sample(problem, N=64, calc_second_order=False, seed=cfg.seed)
    log.info("Sobol: evaluating %d design points in emulator", len(samples))

    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    env = EmulatorEnv(cfg, model=model, meta=meta)

    outputs = []
    for i, sample in enumerate(samples):
        action = sample.astype(np.float32)
        env.reset(seed=cfg.seed + 100 + i)
        total_reward = 0.0
        for _ in range(cfg.horizon_quarters):
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        outputs.append(float(total_reward))

    outputs_array = np.array(outputs)
    sobol_result = sobol_analyze(problem, outputs_array, seed=cfg.seed, calc_second_order=False)

    from typing import cast

    names_list: list[str] = cast(list[str], problem["names"])
    S1 = {str(name): float(s1) for name, s1 in zip(names_list, sobol_result["S1"], strict=True)}
    ST = {str(name): float(st) for name, st in zip(names_list, sobol_result["ST"], strict=True)}
    top_driver = max(S1.items(), key=lambda x: x[1])[0]

    return {
        "status": "quick in-process Sobol (N=64)",
        "method": "Sobol (smoke)",
        "S1": S1,
        "ST": ST,
        "top_driver_S1": top_driver,
        "abm_check_spearman_corr": None,
        "note": "full sensitivity run with Morris + ABM cross-check via `make sensitivity`",
        "thresholds_dev": {
            "S1_positive": "all indices ∈ [0, 1]",
            "ST_gte_S1": "ST ≥ S1 within MC error",
            "abm_check_corr": "> 0.7 expected (not computed in smoke)",
        },
    }
