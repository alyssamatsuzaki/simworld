"""Optuna-based policy search: optimize the 4 levers to maximize regulator objective J.

TPE (Tree-structured Parzen Estimator) sampler over the constant-lever action space.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from regworld.environments.emulator_env import EmulatorEnv
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.training.datamodule import ACTION_HIGH, ACTION_LOW
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def run_policy_search(cfg: RegWorldConfig) -> dict[str, object]:
    """Optuna TPE policy search: maximize J in EmulatorEnv."""
    log.info("Starting Optuna policy search (%d trials)", cfg.sensitivity.optuna_trials)

    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    env = EmulatorEnv(cfg, model=model, meta=meta)

    def objective(trial: optuna.Trial) -> float:
        enforcement = trial.suggest_float(
            "enforcement", float(ACTION_LOW[0]), float(ACTION_HIGH[0])
        )
        targeting = trial.suggest_float("targeting", float(ACTION_LOW[1]), float(ACTION_HIGH[1]))
        phase_speed = trial.suggest_float(
            "phase_speed", float(ACTION_LOW[2]), float(ACTION_HIGH[2])
        )
        subsidy = trial.suggest_float("subsidy", float(ACTION_LOW[3]), float(ACTION_HIGH[3]))

        action = np.array([enforcement, targeting, phase_speed, subsidy], dtype=np.float32)
        env.reset(seed=cfg.seed + 30000 + trial.number)
        total_reward = 0.0
        for _ in range(cfg.horizon_quarters):
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return float(total_reward)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=cfg.seed),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=cfg.sensitivity.optuna_trials, show_progress_bar=False)

    best_trial = study.best_trial
    log.info("Optuna best J: %.4f", best_trial.value)
    log.info("Optuna best levers: %s", best_trial.params)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    result = {
        "method": "Optuna TPE",
        "trials": cfg.sensitivity.optuna_trials,
        "best_J": float(best_trial.value) if best_trial.value is not None else 0.0,
        "best_levers": {
            "enforcement": float(best_trial.params["enforcement"]),
            "targeting": float(best_trial.params["targeting"]),
            "phase_speed": float(best_trial.params["phase_speed"]),
            "subsidy": float(best_trial.params["subsidy"]),
        },
        "n_completed_trials": len(completed),
    }
    return result


def save_optuna_best(cfg: RegWorldConfig, result: dict[str, object]) -> Path:
    """Save Optuna best solution to artifacts."""
    sensitivity_dir = Path(cfg.paths.root) / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    out_path = sensitivity_dir / "optuna_best.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Optuna best → %s", out_path)
    return out_path
