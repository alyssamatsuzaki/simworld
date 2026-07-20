"""Gymnasium interface over the trained GraphRSSM (§10 Stage 8, Phase 5 half).

One contract, two worlds: the observation and action spaces come from the same
constructors as :class:`AbmEnv`, so space identity holds by construction — the
property that makes the planning-utility comparison possible. Steps run in
imagination (prior rollout); reward defaults to exact recomputation from the
decoded outcome vector (``emulator.reward_from_outcomes``), with the learned
two-hot reward head behind the flag, so error can be attributed to dynamics vs
reward modelling.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from regworld.models.world_model import ModelState, WorldModel
from regworld.rules import Constants, regulator_reward
from regworld.training.datamodule import aggregate_dim
from regworld.types import RegWorldConfig

from .wrappers import flat_observation_space, regulator_action_space

_HHI_INDEX = 2
_CS_INDEX = 4
_EXIT_INDEX = 5
_AUDIT_INDEX = 6
_PENALTY_INDEX = 7


def _clip_aggregates(agg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clamp decoded aggregates to their physical ranges."""
    out = agg.copy()
    rate_like = np.ones(len(out), dtype=bool)
    rate_like[[_HHI_INDEX, _CS_INDEX]] = False
    out[rate_like] = np.clip(out[rate_like], 0.0, 1.0)
    out[_HHI_INDEX] = np.clip(out[_HHI_INDEX], 0.0, 10_000.0)
    out[_CS_INDEX] = np.clip(out[_CS_INDEX], -1e6, 1e6)
    return out


class EmulatorEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        cfg: RegWorldConfig,
        *,
        model: WorldModel | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_space = regulator_action_space()
        self.observation_space = flat_observation_space(cfg)
        if model is None or meta is None:
            from regworld.training.checkpoint import checkpoint_path, load_checkpoint

            model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
        if model.aggregate_dim != aggregate_dim(cfg):
            raise ValueError(
                f"checkpoint aggregate dim {model.aggregate_dim} does not match "
                f"config ({aggregate_dim(cfg)}); retrain with this profile"
            )
        self.model = model.eval()
        self.meta = meta
        self._n_firms = int(meta["extras"]["n_firms"])
        self._initial = {k: v.float() for k, v in meta["initial"].items()}
        self._generator = torch.Generator()
        self._state: ModelState | None = None
        self._baseline = np.asarray(self._initial["aggregate"].numpy(), dtype=np.float64)
        self._aggregates = self._baseline.copy()
        self._elapsed = 0
        self._cumulative_audits = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------ observation
    def _observation(self) -> NDArray[np.float32]:
        cfg, agg, base = self.cfg, self._aggregates, self._baseline
        const = Constants()
        max_audits = max(cfg.horizon_quarters * const.audit_budget * self._n_firms, 1.0)
        budget_remaining = 1.0 - self._cumulative_audits / max_audits
        cs_scale = max(abs(base[_CS_INDEX]), 1e-9)
        cs_index = (agg[_CS_INDEX] - base[_CS_INDEX]) / cs_scale
        n_sectors = cfg.population.n_sectors
        obs = np.concatenate(
            [
                np.array(
                    [
                        agg[0],
                        agg[1],
                        agg[_HHI_INDEX],
                        agg[_HHI_INDEX] - base[_HHI_INDEX],
                        agg[3],
                        np.clip(cs_index, -10.0, 10.0),
                        agg[_EXIT_INDEX],
                        np.clip(budget_remaining, 0.0, 1.0),
                        np.clip(self._elapsed / max(cfg.horizon_quarters, 1), 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
                agg[8 : 8 + n_sectors].astype(np.float32),
                agg[8 + n_sectors : 8 + n_sectors + 10].astype(np.float32),
                np.array([agg[_AUDIT_INDEX], np.clip(agg[_PENALTY_INDEX], 0.0, 1.0)], np.float32),
                self._last_action,
            ]
        )
        space = cast(gym.spaces.Box, self.observation_space)
        return cast(
            NDArray[np.float32],
            np.clip(obs, space.low, space.high).astype(np.float32, copy=False),
        )

    def _alive_count(self) -> float:
        return max((1.0 - float(self._aggregates[_EXIT_INDEX])) * self._n_firms, 1.0)

    def _collapsed(self) -> bool:
        const = Constants()
        max_audits = max(self.cfg.horizon_quarters * const.audit_budget * self._n_firms, 1.0)
        budget_remaining = 1.0 - self._cumulative_audits / max_audits
        return bool(
            self._aggregates[_EXIT_INDEX] > 0.40
            or (self._elapsed > 12 and self._aggregates[0] < 0.05 and budget_remaining <= 0.0)
        )

    # ------------------------------------------------------------------- API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        env_seed = self.cfg.seed if seed is None else seed
        self._generator.manual_seed(env_seed)
        self._state = self.model.initial_state(
            self._initial["firm"].unsqueeze(0),
            self._initial["segment"].unsqueeze(0),
            self._initial["aggregate"].unsqueeze(0),
            self._generator,
        )
        self._aggregates = self._baseline.copy()
        self._elapsed = 0
        self._cumulative_audits = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)
        return self._observation(), {"seed": env_seed, "backend": "emulator"}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        action_box = cast(gym.spaces.Box, self.action_space)
        self._last_action = np.clip(action, action_box.low, action_box.high).astype(
            np.float32, copy=False
        )
        action_t = torch.as_tensor(self._last_action, dtype=torch.float32).unsqueeze(0)
        self._state, decoded = self.model.imagine_step(self._state, action_t, self._generator)
        self._aggregates = _clip_aggregates(decoded.aggregates[0].numpy().astype(np.float64))
        self._elapsed += 1
        self._cumulative_audits += float(self._aggregates[_AUDIT_INDEX]) * self._alive_count()
        if self.cfg.emulator.reward_from_outcomes:
            from regworld.training.datamodule import aggregate_to_outcome

            weights = tuple(
                float(getattr(self.cfg.objective, name))
                for name in ("w_c", "w_h", "w_s", "w_e", "w_t", "w_x")
            )
            reward = regulator_reward(
                aggregate_to_outcome(self._aggregates, self._n_firms),
                aggregate_to_outcome(self._baseline, self._n_firms),
                cast(tuple[float, float, float, float, float, float], weights),
                Constants(),
                self._n_firms,
            )
        else:
            reward = float(decoded.reward[0])
        terminated = self._collapsed()
        truncated = self._elapsed >= self.cfg.horizon_quarters and not terminated
        info = {
            "elapsed_quarters": self._elapsed,
            "continue_prob": float(decoded.continue_prob[0]),
            "backend": "emulator",
        }
        return self._observation(), float(reward), terminated, truncated, info
