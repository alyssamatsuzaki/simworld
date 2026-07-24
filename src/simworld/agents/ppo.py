"""Stage 10b — SB3 PPO trained inside ``EmulatorEnv``: the control group.

The plan's own words: "It is the control group, not the experiment." Its
number exists so the Dreamer agent (Stage 10c) has something to beat. The
checkpoint is loaded once and shared read-only across every vec-env worker
(SB3's ``DummyVecEnv`` keeps everything in one process, so "shared" here
literally means "the same Python object", not a re-parsed copy per worker).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from simworld.agents.registry import load_checkpoint_compat
from simworld.environments.emulator_env import EmulatorEnv
from simworld.models.world_model import WorldModel
from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)


@dataclass
class PpoResult:
    checkpoint: Path
    summary: Path
    metrics: dict[str, float] = field(default_factory=dict)


def _make_env(
    cfg: SimWorldConfig, model: WorldModel, meta: dict[str, Any], seed: int
) -> Callable[[], EmulatorEnv]:
    def _init() -> EmulatorEnv:
        env = EmulatorEnv(cfg, model=model, meta=meta)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def train_ppo(
    cfg: SimWorldConfig,
    *,
    model: WorldModel | None = None,
    meta: dict[str, Any] | None = None,
) -> PpoResult:
    """Train SB3 PPO on a vec-env of ``EmulatorEnv`` copies sharing one model."""
    if model is None or meta is None:
        model, meta = load_checkpoint_compat(cfg)
    model = model.eval()

    n_envs = max(1, cfg.rl.n_envs)
    vec_env = DummyVecEnv([_make_env(cfg, model, meta, cfg.seed + i) for i in range(n_envs)])
    vec_env.seed(cfg.seed)

    n_steps = max(8, min(2048, cfg.rl.total_timesteps // n_envs))
    buffer_size = n_steps * n_envs
    batch_size = max(2, min(64, buffer_size))

    agent = PPO(
        "MlpPolicy",
        vec_env,
        seed=cfg.seed,
        n_steps=n_steps,
        batch_size=batch_size,
        device="cpu",
        verbose=0,
    )
    agent.learn(total_timesteps=cfg.rl.total_timesteps, progress_bar=False)

    out_dir = Path(cfg.paths.root) / "rl" / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "model"))
    checkpoint = out_dir / "model.zip"

    eval_env = EmulatorEnv(cfg, model=model, meta=meta)
    n_eval_episodes = max(1, min(5, n_envs))
    mean_reward, std_reward = evaluate_policy(
        agent,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        warn=False,
        return_episode_rewards=False,
    )
    eval_env.close()
    vec_env.close()

    metrics = {
        "mean_episode_reward": float(cast(float, mean_reward)),
        "std_episode_reward": float(cast(float, std_reward)),
        "total_timesteps": float(cfg.rl.total_timesteps),
        "n_envs": float(n_envs),
        "n_steps": float(n_steps),
    }
    summary = out_dir / "train_summary.json"
    summary.write_text(json.dumps(metrics, indent=2))
    log.info("PPO trained: mean_reward=%.4f +/- %.4f -> %s", mean_reward, std_reward, checkpoint)
    return PpoResult(checkpoint=checkpoint, summary=summary, metrics=metrics)
