"""Stage 10c — a Dreamer-style latent actor-critic trained purely on imagined
rollouts of the trained GraphRSSM (Phase 5). The actor and critic never see a
transition that did not come out of ``WorldModel.imagine_step``: the whole
point of the exploitation-gap test is that this agent has literally never
been told what the true ABM does.

``WorldModel.imagine_step`` is decorated ``@torch.no_grad()`` — it is the same
call ``EmulatorEnv`` steps with, and it has to stay a cheap, ungraphed
inference call there. That forecloses the textbook Dreamer recipe of
backpropagating the actor's objective straight through the dynamics
(reparameterized/pathwise gradients through the model). This module trains
the actor instead with a score-function (REINFORCE) estimator against a
learned critic baseline: still a latent actor-critic trained end-to-end on
imagined trajectories with lambda-returns and a model-predicted continuation
discount, just without a pathwise gradient through the world model. This is
the plan's own documented fallback ("hand-roll the loop in plain PyTorch")
applied to the model's API rather than to a library's.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from simworld.agents.registry import load_checkpoint_compat
from simworld.environments.emulator_env import EmulatorEnv
from simworld.environments.wrappers import flat_observation_space, regulator_action_space
from simworld.models.world_model import WorldModel
from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)

_GAMMA = 0.99
_LAMBDA = 0.95
_ENTROPY_COEF = 1.0e-3
_MAX_HORIZON = 15


@dataclass
class DreamerResult:
    actor_path: Path
    meta_path: Path
    summary: Path
    metrics: dict[str, float] = field(default_factory=dict)


def normalize_obs(obs: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Map a raw flat observation into roughly [0, 1] using the space bounds."""
    span = (high - low).clamp(min=1.0e-6)
    return (obs - low) / span


def squash_action(u: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """tanh-squash an unconstrained sample into the regulator's action bounds."""
    unit = (torch.tanh(u) + 1.0) * 0.5
    return low + unit * (high - low)


class SquashedGaussianActor(nn.Module):
    """Normalized observation -> bounded action distribution parameters."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs_norm)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(-5.0, 2.0)
        return mean, log_std


class Critic(nn.Module):
    """Normalized observation -> scalar imagined-return value estimate."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.net(obs_norm).squeeze(-1)
        return value


def _budget(cfg: SimWorldConfig) -> tuple[int, int, int]:
    """(batch_size, horizon, n_updates), bounded so smoke stays seconds-long."""
    batch_size = max(2, min(8, cfg.rl.n_envs))
    horizon = max(2, min(_MAX_HORIZON, cfg.emulator.imag_horizon, cfg.horizon_quarters))
    n_updates = max(2, min(50, cfg.rl.total_timesteps // 500))
    return batch_size, horizon, n_updates


def _lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """(H, B) rewards/values/continues + (B,) bootstrap -> (H, B) TD(lambda) returns."""
    horizon = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    next_return = bootstrap
    for t in reversed(range(horizon)):
        next_value = values[t + 1] if t + 1 < horizon else bootstrap
        next_return = rewards[t] + gamma * continues[t] * (
            (1.0 - lam) * next_value + lam * next_return
        )
        returns[t] = next_return
    return returns


def _imagination_rollout(
    envs: list[EmulatorEnv],
    actor: SquashedGaussianActor,
    critic: Critic,
    *,
    horizon: int,
    obs_low: torch.Tensor,
    obs_high: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    reset_seeds: list[int],
) -> dict[str, torch.Tensor]:
    """Roll ``horizon`` imagined steps across ``envs`` (never the true ABM)."""
    batch = len(envs)
    obs_np = np.stack(
        [env.reset(seed=seed)[0] for env, seed in zip(envs, reset_seeds, strict=True)]
    )
    active = torch.ones(batch)

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continues: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for _step in range(horizon):
        mask_t = active.clone()
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        obs_norm = normalize_obs(obs, obs_low, obs_high)
        values.append(critic(obs_norm))
        mean, log_std = actor(obs_norm)
        dist = torch.distributions.Normal(mean, log_std.exp())
        # A detached sample: the REINFORCE gradient comes from log_prob's
        # explicit dependence on (mean, std), not from differentiating
        # through the sampling path (that would double-count / cancel it).
        u = dist.sample()
        action = squash_action(u, action_low, action_high)
        log_probs.append(dist.log_prob(u).sum(-1))
        entropies.append(dist.entropy().sum(-1))

        action_np = action.numpy().astype(np.float32)
        step_reward = np.zeros(batch, dtype=np.float32)
        step_continue = np.zeros(batch, dtype=np.float32)
        for i, env in enumerate(envs):
            if mask_t[i].item() <= 0.5:
                continue
            nobs, reward, terminated, truncated, _info = env.step(action_np[i])
            obs_np[i] = nobs
            step_reward[i] = float(reward)
            step_continue[i] = 0.0 if terminated else 1.0
            if terminated or truncated:
                active[i] = 0.0

        rewards.append(torch.as_tensor(step_reward))
        continues.append(torch.as_tensor(step_continue))
        masks.append(mask_t)

    final_obs = torch.as_tensor(obs_np, dtype=torch.float32)
    bootstrap = critic(normalize_obs(final_obs, obs_low, obs_high)).detach()
    return {
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "rewards": torch.stack(rewards),
        "continues": torch.stack(continues),
        "masks": torch.stack(masks),
        "values": torch.stack(values),
        "bootstrap": bootstrap,
    }


def compute_losses(rollout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Actor (REINFORCE + entropy bonus) and critic (lambda-return regression) losses."""
    returns = _lambda_returns(
        rollout["rewards"],
        rollout["values"].detach(),
        rollout["continues"],
        rollout["bootstrap"],
        gamma=_GAMMA,
        lam=_LAMBDA,
    )
    masks = rollout["masks"]
    n_active = masks.sum().clamp(min=1.0)
    advantage = (returns - rollout["values"].detach()) * masks
    actor_loss = -(rollout["log_probs"] * advantage).sum() / n_active
    actor_loss = actor_loss - _ENTROPY_COEF * (rollout["entropies"] * masks).sum() / n_active
    critic_loss = (((rollout["values"] - returns.detach()) ** 2) * masks).sum() / n_active
    mean_imagined_return = (returns * masks).sum() / n_active
    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "mean_imagined_return": mean_imagined_return,
    }


def train_dreamer(
    cfg: SimWorldConfig,
    *,
    model: WorldModel | None = None,
    meta: dict[str, Any] | None = None,
) -> DreamerResult:
    """Train the latent actor-critic entirely on ``EmulatorEnv`` imagination."""
    torch.manual_seed(cfg.seed + 7)
    if model is None or meta is None:
        model, meta = load_checkpoint_compat(cfg)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    obs_space = flat_observation_space(cfg)
    action_space = regulator_action_space()
    obs_dim = int(obs_space.shape[0])
    obs_low = torch.as_tensor(obs_space.low, dtype=torch.float32)
    obs_high = torch.as_tensor(obs_space.high, dtype=torch.float32)
    action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(action_space.high, dtype=torch.float32)

    actor = SquashedGaussianActor(obs_dim, action_dim=4)
    critic = Critic(obs_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3.0e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1.0e-3)

    batch_size, horizon, n_updates = _budget(cfg)
    envs = [EmulatorEnv(cfg, model=model, meta=meta) for _ in range(batch_size)]
    rng = np.random.default_rng(cfg.seed + 4242)

    history: list[dict[str, float]] = []
    try:
        for update in range(n_updates):
            seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=batch_size)]
            rollout = _imagination_rollout(
                envs,
                actor,
                critic,
                horizon=horizon,
                obs_low=obs_low,
                obs_high=obs_high,
                action_low=action_low,
                action_high=action_high,
                reset_seeds=seeds,
            )
            losses = compute_losses(rollout)

            actor_opt.zero_grad(set_to_none=True)
            losses["actor_loss"].backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_opt.step()

            critic_opt.zero_grad(set_to_none=True)
            losses["critic_loss"].backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_opt.step()

            snapshot = {key: float(value.item()) for key, value in losses.items()}
            snapshot["update"] = float(update)
            history.append(snapshot)
            log.info(
                "dreamer update %d/%d actor %.4f critic %.4f imagined_return %.4f",
                update,
                n_updates,
                snapshot["actor_loss"],
                snapshot["critic_loss"],
                snapshot["mean_imagined_return"],
            )
    finally:
        for env in envs:
            env.close()

    out_dir = Path(cfg.paths.root) / "rl" / "dreamer"
    out_dir.mkdir(parents=True, exist_ok=True)
    actor_path = out_dir / "actor.pt"
    torch.save(actor.state_dict(), actor_path)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"obs_dim": obs_dim, "action_dim": 4, "hidden_dim": actor.hidden_dim}, indent=2)
    )

    metrics = {
        "final_actor_loss": history[-1]["actor_loss"],
        "final_critic_loss": history[-1]["critic_loss"],
        "final_mean_imagined_return": history[-1]["mean_imagined_return"],
        "n_updates": float(n_updates),
        "batch_size": float(batch_size),
        "horizon": float(horizon),
    }
    summary = out_dir / "train_summary.json"
    summary.write_text(json.dumps({"metrics": metrics, "history": history}, indent=2))
    log.info("dreamer actor trained -> %s", actor_path)
    return DreamerResult(
        actor_path=actor_path, meta_path=meta_path, summary=summary, metrics=metrics
    )
