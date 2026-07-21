"""Stage 10 policy seam: one lookup for every scripted, learned, or random policy.

Both this stage's own planning-utility gate and the ensemble stage (Stage 11)
depend on exactly this contract, so it stays intentionally small: a policy is
nothing more than ``obs (obs_dim,) -> action (4,)``. Static levers, a seeded
random policy, and any trained artifact under ``artifacts/rl/`` are all wrapped
to the same shape so the evaluation code never has to know which one it is
calling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from regworld.abm.policies import STATIC_POLICIES
from regworld.environments.wrappers import flat_observation_space, regulator_action_space
from regworld.models.world_model import WorldModel
from regworld.rules import PolicyLevers
from regworld.training.checkpoint import checkpoint_path, load_checkpoint
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

PolicyFn = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def load_checkpoint_compat(cfg: RegWorldConfig) -> tuple[WorldModel, dict[str, Any]]:
    """``load_checkpoint`` plus a defensive backfill of ``extras["n_firms"]``.

    ``EmulatorEnv`` requires ``meta["extras"]["n_firms"]``, but a checkpoint
    saved before that field existed only carries the initial firm/segment/
    aggregate frame. When it is missing, derive it from the initial firm
    frame's leading dimension instead of failing every downstream consumer.
    """
    model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    extras = dict(meta.get("extras", {}))
    if "n_firms" not in extras:
        n_firms = int(meta["initial"]["firm"].shape[0])
        log.warning(
            "emulator checkpoint extras missing 'n_firms'; backfilling %d from the "
            "initial firm frame",
            n_firms,
        )
        extras["n_firms"] = n_firms
        meta = {**meta, "extras": extras}
    return model, meta


def _ppo_checkpoint(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "rl" / "ppo" / "model.zip"


def _dreamer_checkpoint(cfg: RegWorldConfig) -> tuple[Path, Path]:
    root = Path(cfg.paths.root) / "rl" / "dreamer"
    return root / "actor.pt", root / "meta.json"


def _constant_policy(
    levers: PolicyLevers,
    action_space_low: NDArray[np.float32],
    action_space_high: NDArray[np.float32],
) -> PolicyFn:
    action = np.clip(levers.as_array().astype(np.float32), action_space_low, action_space_high)

    def policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
        del observation
        return action.copy()

    return policy


def _random_policy(cfg: RegWorldConfig) -> PolicyFn:
    space = regulator_action_space()
    rng = np.random.default_rng(cfg.seed)

    def policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
        del observation
        return rng.uniform(space.low, space.high).astype(np.float32)

    return policy


def _ppo_policy(cfg: RegWorldConfig) -> PolicyFn:
    checkpoint = _ppo_checkpoint(cfg)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"rl_ppo policy artifact not found at {checkpoint}; "
            "run `uv run python scripts/train_rl.py` (Stage 10) first"
        )
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")

    def policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
        action, _state = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return policy


def _dreamer_policy(cfg: RegWorldConfig) -> PolicyFn:
    from regworld.agents.dreamer import SquashedGaussianActor, normalize_obs, squash_action

    actor_path, meta_path = _dreamer_checkpoint(cfg)
    if not actor_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"rl_dreamer policy artifact not found at {actor_path}; "
            "run `uv run python scripts/train_rl.py` (Stage 10, rl.train_dreamer=true) first"
        )
    meta = json.loads(meta_path.read_text())
    actor = SquashedGaussianActor(
        obs_dim=int(meta["obs_dim"]),
        action_dim=int(meta["action_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
    )
    actor.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True))
    actor.eval()

    obs_space = flat_observation_space(cfg)
    action_space = regulator_action_space()
    obs_low = torch.as_tensor(obs_space.low, dtype=torch.float32)
    obs_high = torch.as_tensor(obs_space.high, dtype=torch.float32)
    action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(action_space.high, dtype=torch.float32)

    def policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.no_grad():
            obs_t = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            obs_norm = normalize_obs(obs_t, obs_low, obs_high)
            mean, _log_std = actor(obs_norm)
            action = squash_action(mean, action_low, action_high)
        return action.squeeze(0).numpy().astype(np.float32)

    return policy


def load_policy(cfg: RegWorldConfig, name: str) -> PolicyFn:
    """Resolve a policy name to a callable ``obs -> action``.

    ``name`` may be any key of ``STATIC_POLICIES`` (constant levers), the
    literal ``"random"`` (seeded uniform draw in the action bounds), or one of
    the learned artifact names ``"rl_ppo"`` / ``"rl_dreamer"``. Raises a clear
    error for a learned name whose artifact has not been trained yet.
    """
    action_space = regulator_action_space()
    if name in STATIC_POLICIES:
        return _constant_policy(STATIC_POLICIES[name], action_space.low, action_space.high)
    if name == "random":
        return _random_policy(cfg)
    if name == "rl_ppo":
        return _ppo_policy(cfg)
    if name == "rl_dreamer":
        return _dreamer_policy(cfg)
    raise ValueError(f"unknown policy {name!r}; available: {available_policies(cfg)}")


def available_policies(cfg: RegWorldConfig) -> list[str]:
    """Static policy names + ``random``, plus any learned policy already trained."""
    names = [*STATIC_POLICIES.keys(), "random"]
    if _ppo_checkpoint(cfg).is_file():
        names.append("rl_ppo")
    actor_path, meta_path = _dreamer_checkpoint(cfg)
    if actor_path.is_file() and meta_path.is_file():
        names.append("rl_dreamer")
    return names
