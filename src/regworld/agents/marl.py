"""Stage 10d — strategic firms actually learn, and C6 becomes computable.

**DEGRADED, on purpose.** PLAN.md Stage 10d asks for RLlib multi-agent PPO but
guardrail 11 declares RLlib non-gating and sanctions the fallback this module
implements: *hand-rolled independent PPO with iterated best response — wrap the
parallel env into N single-agent views with the other agents frozen, train each
with SB3 PPO, iterate rounds.* No RLlib, no new dependency.

Three pieces live here.

``SingleAgentView``
    A Gymnasium view over :class:`~regworld.environments.marl_env.RegulationMARLEnv`
    that exposes exactly one *ego* agent; every other agent acts from a frozen
    :class:`PolicyBook`. Observations are flattened (the firm agents observe a
    ``Dict``) and min-max normalized into ``[0, 1]`` so a tiny PPO run is not
    asked to condition on a raw HHI in the thousands. Gymnasium five-tuple
    semantics are inherited from the parallel env: ``truncated`` at the horizon,
    ``terminated`` only on systemic collapse or the ego firm's own exit.

``train_marl``
    The iterated-best-response loop. Firms share one policy (PLAN.md 10d:
    "parameter sharing across firms optional") and the ego rotates across
    ``firm_0 … firm_{K-1}`` from episode to episode, so the shared policy is
    fitted on every strategic firm's local view rather than only the largest
    one. Each round fits the firm policy against the frozen regulator, then the
    regulator against the freshly frozen firm policy.

``compare_c6``
    The claim-C6 ablation: the headline C5 quantities (terminal compliance,
    HHI, delta-HHI, backfire probability) under strategic-firm MARL versus the
    rule-based-firm baseline, on the *same* episode seeds, with 95% CIs — so
    PLAN.md's verdict rule ("if the 95% credible intervals overlap, report that
    MARL did not change the conclusion") is mechanically checkable rather than
    a judgement call. The baseline arm is the same environment driven with
    all-zero firm actions, which
    :func:`regworld.abm.model.strategic_controls_from_actions` maps exactly onto
    ``StrategicControls.neutral`` — i.e. the calibrated rule-based firms.

This module must never import :mod:`regworld.dgp` or read the answer key.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO

from regworld.abm.policies import STATIC_POLICIES, levers_from_config
from regworld.agents.planning import RolloutStats
from regworld.environments.abm_env import ModelFactory
from regworld.environments.marl_env import AgentObservation, RegulationMARLEnv
from regworld.rules import PolicyLevers, QuarterOutcome, backfire
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

REGULATOR = "regulator_0"
BACKEND = "ippo_iterated_best_response"
SCHEMA = "regworld.marl.c6_comparison.v1"

#: Best-response rounds per profile. PLAN.md 10d prescribes three; the smoke
#: profile buys the wall-clock budget back by dropping to two.
IBR_ROUNDS: dict[str, int] = {"smoke": 2, "dev": 3, "full": 3}

#: Floor on one best-response fit so PPO always sees at least a couple of
#: rollout buffers even when ``cfg.rl.marl_timesteps`` is tiny.
MIN_FIT_TIMESTEPS = 128

#: Per-episode quantities recorded for every arm. ``backfire_rate`` is 0/1 per
#: episode, so its arm mean *is* the backfire probability.
C6_METRICS: tuple[str, ...] = (
    "terminal_compliance",
    "hhi",
    "delta_hhi",
    "backfire_rate",
    "regulator_return",
)

#: The subset of :data:`C6_METRICS` that C5 actually headlines; the verdict is
#: read off these and these only.
C6_HEADLINE_METRICS: tuple[str, ...] = (
    "terminal_compliance",
    "hhi",
    "delta_hhi",
    "backfire_rate",
)

AgentActionFn = Callable[[AgentObservation], NDArray[np.float32]]


# --------------------------------------------------------------------------- #
# observation encoding                                                        #
# --------------------------------------------------------------------------- #


def _space_bounds(space: spaces.Space[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Flat ``(low, high)`` for a ``Box`` or a ``Dict`` of ``Box``es."""
    if isinstance(space, spaces.Box):
        return space.low.ravel().astype(np.float32), space.high.ravel().astype(np.float32)
    if isinstance(space, spaces.Dict):
        # gymnasium's Dict keeps its keys sorted, which is the order _flatten uses.
        pairs = [_space_bounds(space.spaces[key]) for key in space.spaces]
        return (
            np.concatenate([low for low, _ in pairs]),
            np.concatenate([high for _, high in pairs]),
        )
    raise TypeError(f"unsupported observation space {type(space).__name__}")


def _flatten(observation: AgentObservation) -> NDArray[np.float32]:
    if isinstance(observation, dict):
        return np.concatenate(
            [np.asarray(observation[key], dtype=np.float32).ravel() for key in sorted(observation)]
        )
    return np.asarray(observation, dtype=np.float32).ravel()


class ObservationEncoder:
    """Flatten one agent's observation and min-max it into ``[0, 1]^d``."""

    def __init__(self, space: spaces.Space[Any]) -> None:
        low, high = _space_bounds(space)
        span = high - low
        self.low = low
        self.span = np.where(span > 1e-9, span, np.float32(1.0)).astype(np.float32)
        self.space = spaces.Box(0.0, 1.0, shape=(low.size,), dtype=np.float32)

    def __call__(self, observation: AgentObservation) -> NDArray[np.float32]:
        flat = _flatten(observation)
        return cast(
            NDArray[np.float32],
            np.clip((flat - self.low) / self.span, 0.0, 1.0).astype(np.float32, copy=False),
        )


# --------------------------------------------------------------------------- #
# frozen-policy book                                                          #
# --------------------------------------------------------------------------- #


def constant_action_fn(action: NDArray[np.float32]) -> AgentActionFn:
    """An agent that plays one fixed action forever (the frozen round-0 opponent)."""
    fixed = np.asarray(action, dtype=np.float32).copy()

    def act(observation: AgentObservation) -> NDArray[np.float32]:
        del observation
        return fixed.copy()

    return act


def sb3_action_fn(agent: PPO, encoder: ObservationEncoder) -> AgentActionFn:
    """Freeze a trained SB3 policy into an ``observation -> action`` callable."""

    def act(observation: AgentObservation) -> NDArray[np.float32]:
        action, _state = agent.predict(encoder(observation), deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return act


@dataclass(frozen=True)
class PolicyBook:
    """Who plays what: one regulator policy, one shared strategic-firm policy."""

    regulator: AgentActionFn
    firm: AgentActionFn

    def act(self, agent: str, observation: AgentObservation) -> NDArray[np.float32]:
        return self.regulator(observation) if agent == REGULATOR else self.firm(observation)


def rule_based_firm_fn() -> AgentActionFn:
    """The C6 control arm: zero strategic action == ``StrategicControls.neutral``."""
    return constant_action_fn(np.zeros(3, dtype=np.float32))


def reference_levers(cfg: RegWorldConfig) -> PolicyLevers:
    """The regulator held fixed in both C6 arms, so only firm behaviour varies."""
    try:
        return levers_from_config(cfg.policy)
    except ValueError:
        fallback = STATIC_POLICIES["phased_targeted"]
        log.warning(
            "cfg.policy %r is learned; C6 arms use the static %r levers instead",
            cfg.policy.name,
            "phased_targeted",
        )
        return fallback


def reference_book(cfg: RegWorldConfig) -> PolicyBook:
    """Rule-based firms under the reference regulator — the C6 baseline arm."""
    return PolicyBook(
        regulator=constant_action_fn(reference_levers(cfg).as_array().astype(np.float32)),
        firm=rule_based_firm_fn(),
    )


# --------------------------------------------------------------------------- #
# the single-agent view                                                       #
# --------------------------------------------------------------------------- #


def episode_seed(base_seed: int, episode: int) -> int:
    """Deterministic, distinct model seed for one training/eval episode."""
    sequence = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, int(episode) & 0xFFFFFFFF])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class SingleAgentView(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """One agent of the Stage-9 parallel env, everyone else frozen at ``book``."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        cfg: RegWorldConfig,
        role: str,
        book: PolicyBook,
        *,
        model_factory: ModelFactory | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if role not in ("regulator", "firm"):
            raise ValueError(f"role must be 'regulator' or 'firm', got {role!r}")
        n_strategic = int(cfg.env.n_strategic_firms)
        if role == "firm" and n_strategic < 1:
            raise ValueError("cfg.env.n_strategic_firms must be >= 1 to train a firm policy")
        self.cfg = cfg
        self.role = role
        self.book = book
        self._env = RegulationMARLEnv(cfg, model_factory=model_factory)
        self._ego_pool: tuple[str, ...] = (
            (REGULATOR,) if role == "regulator" else tuple(f"firm_{i}" for i in range(n_strategic))
        )
        # every firm shares one observation/action space, so pool[0] defines both
        self.encoder = ObservationEncoder(self._env.observation_space(self._ego_pool[0]))
        self.observation_space = self.encoder.space
        self.action_space = self._env.action_space(self._ego_pool[0])
        self._base_seed = int(seed)
        self._episode = 0
        self.ego = self._ego_pool[0]
        self._observations: dict[str, AgentObservation] = {}

    @property
    def ego_pool(self) -> tuple[str, ...]:
        return self._ego_pool

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            # An explicit seed restarts the episode stream, so reset(seed=s) twice
            # replays the same episode *and* the same ego (Gymnasium's checker
            # requires it, and reproducibility wants it).
            self._base_seed = int(seed)
            self._episode = 0
        self.ego = self._ego_pool[self._episode % len(self._ego_pool)]
        model_seed = episode_seed(self._base_seed, self._episode)
        self._episode += 1
        self._observations, infos = self._env.reset(model_seed)
        return self.encoder(self._observations[self.ego]), dict(infos.get(self.ego, {}))

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if not self._observations:
            raise RuntimeError("reset() must be called before step()")
        if self.ego not in self._env.agents:
            raise RuntimeError(f"{self.ego} is already done; reset() before stepping again")
        ego_action = np.asarray(action, dtype=np.float32)
        joint = {
            agent: ego_action
            if agent == self.ego
            else self.book.act(agent, self._observations[agent])
            for agent in self._env.agents
        }
        observations, rewards, terminations, truncations, infos = self._env.step(joint)
        self._observations = observations
        ego = self.ego
        return (
            self.encoder(observations[ego]),
            float(rewards[ego]),
            bool(terminations[ego]),
            bool(truncations[ego]),
            dict(infos.get(ego, {})),
        )

    def close(self) -> None:
        self._env.close()


# --------------------------------------------------------------------------- #
# iterated best response                                                      #
# --------------------------------------------------------------------------- #


def _rollout_length(timesteps: int) -> int:
    """Largest power-of-two buffer (<=256) that still fills at least twice."""
    for candidate in (256, 128, 64, 32):
        if timesteps >= 2 * candidate:
            return candidate
    return 32


def fit_best_response(
    view: SingleAgentView, timesteps: int, seed: int
) -> tuple[PPO, dict[str, float]]:
    """One PPO best-response fit against the frozen opponents inside ``view``."""
    n_steps = _rollout_length(timesteps)
    agent = PPO(
        "MlpPolicy",
        view,
        seed=seed,
        n_steps=n_steps,
        batch_size=32,
        n_epochs=4,
        device="cpu",
        verbose=0,
    )
    agent.learn(total_timesteps=timesteps, progress_bar=False)
    episode_returns = [float(entry["r"]) for entry in (agent.ep_info_buffer or [])]
    stats = {
        "requested_timesteps": float(timesteps),
        "actual_timesteps": float(agent.num_timesteps),
        "n_steps": float(n_steps),
        "n_episodes": float(len(episode_returns)),
        "mean_episode_reward": float(np.mean(episode_returns)) if episode_returns else float("nan"),
    }
    return agent, stats


# --------------------------------------------------------------------------- #
# C6: strategic firms vs rule-based firms                                     #
# --------------------------------------------------------------------------- #


def _outcomes(env: RegulationMARLEnv) -> tuple[QuarterOutcome, QuarterOutcome]:
    """The env's terminal and quarter-0 outcome rows, in natural units.

    ``RegulationMARLEnv``'s Parallel API only publishes clipped/normalized
    observations, so the natural-unit rows come off its ``_outcome`` /
    ``_baseline`` attributes — the same access pattern ``ensemble.cube`` uses
    for ``EmulatorEnv._aggregates``.
    """
    terminal, baseline = env._outcome, env._baseline
    if terminal is None or baseline is None:
        raise RuntimeError("reset() must be called before reading outcomes")
    return terminal, baseline


def rollout_arm(env: RegulationMARLEnv, book: PolicyBook, seed: int) -> dict[str, float]:
    """One full episode with every agent driven by ``book``; terminal C5 row out."""
    observations, _infos = env.reset(seed)
    regulator_return = 0.0
    quarters = 0
    while env.agents:
        actions = {agent: book.act(agent, observations[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, _infos = env.step(actions)
        regulator_return += float(rewards.get(REGULATOR, 0.0))
        quarters += 1
        if terminations.get(REGULATOR, False) or truncations.get(REGULATOR, False):
            break
    terminal, baseline = _outcomes(env)
    return {
        "terminal_compliance": float(terminal.compliance_rate),
        "hhi": float(terminal.hhi),
        "delta_hhi": float(terminal.hhi - baseline.hhi),
        "backfire_rate": float(backfire(terminal, baseline)),
        "regulator_return": regulator_return,
        "quarters": float(quarters),
        "collapsed": float(quarters < env.cfg.horizon_quarters),
    }


def evaluate_arm(
    cfg: RegWorldConfig,
    book: PolicyBook,
    seeds: Sequence[int],
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, list[float]]:
    """Roll ``book`` over ``seeds`` and transpose into metric -> episode values."""
    env = RegulationMARLEnv(cfg, model_factory=model_factory)
    try:
        rows = [rollout_arm(env, book, seed) for seed in seeds]
    finally:
        env.close()
    return {key: [row[key] for row in rows] for key in rows[0]} if rows else {}


def _stats(values: Sequence[float]) -> RolloutStats:
    array = np.asarray(values, dtype=np.float64)
    return RolloutStats(
        mean=float(array.mean()) if array.size else float("nan"),
        std=float(array.std(ddof=0)) if array.size else float("nan"),
        n=int(array.size),
        returns=tuple(float(value) for value in array),
    )


def _stats_json(stats: RolloutStats) -> dict[str, Any]:
    low, high = stats.ci95()
    return {"mean": stats.mean, "std": stats.std, "n": stats.n, "ci95": [low, high]}


def _paired_diff_ci(baseline: Sequence[float], strategic: Sequence[float]) -> list[float]:
    """95% CI on the per-episode difference (arms share their seed grid)."""
    diff = np.asarray(strategic, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if diff.size == 0:
        return [float("nan"), float("nan")]
    se = float(diff.std(ddof=0)) / max(float(np.sqrt(diff.size)), 1.0)
    mean = float(diff.mean())
    return [mean - 1.96 * se, mean + 1.96 * se]


def compare_arms(
    baseline: dict[str, list[float]], strategic: dict[str, list[float]]
) -> dict[str, Any]:
    """Per-metric overlap test between two arms evaluated on the same seeds."""
    metrics: dict[str, Any] = {}
    for name in C6_METRICS:
        base_values, strat_values = baseline.get(name, []), strategic.get(name, [])
        base_stats, strat_stats = _stats(base_values), _stats(strat_values)
        base_low, base_high = base_stats.ci95()
        strat_low, strat_high = strat_stats.ci95()
        overlap = bool(base_low <= strat_high and strat_low <= base_high)
        metrics[name] = {
            "baseline": _stats_json(base_stats),
            "strategic": _stats_json(strat_stats),
            "diff": strat_stats.mean - base_stats.mean,
            "diff_ci95": _paired_diff_ci(base_values, strat_values),
            "ci_overlap": overlap,
            "changed": not overlap,
        }
    changed = [name for name in C6_HEADLINE_METRICS if metrics[name]["changed"]]
    return {
        "baseline_arm": "rule_based",
        "strategic_arm": "strategic",
        "headline_metrics": list(C6_HEADLINE_METRICS),
        "ci_method": "normal approximation on the per-episode mean (mean +/- 1.96 * SE)",
        "metrics": metrics,
        "changed_metrics": changed,
        "any_changed": bool(changed),
        "verdict": ("MARL changed C5" if changed else "MARL did not change the conclusion"),
    }


def marl_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "marl"


def comparison_path(cfg: RegWorldConfig) -> Path:
    """The artifact ``evaluation.report`` reads for claim C6."""
    return marl_dir(cfg) / "c6_comparison.json"


def compare_c6(
    cfg: RegWorldConfig,
    book: PolicyBook,
    *,
    marl_regulator: AgentActionFn | None = None,
    model_factory: ModelFactory | None = None,
    n_episodes: int | None = None,
) -> dict[str, Any]:
    """Evaluate C5's headline quantities with and without strategic firms.

    ``book`` carries the trained shared firm policy. Both headline arms hold the
    regulator at :func:`reference_levers`, so the only thing that varies is
    whether the ten largest firms optimize. ``marl_regulator``, when given,
    adds a third (supplementary) arm in which the regulator is the trained MARL
    policy as well.
    """
    episodes = int(n_episodes if n_episodes is not None else max(8, cfg.eval.n_dist_rollouts))
    seeds = [episode_seed(cfg.seed + 7919, index) for index in range(episodes)]
    regulator = constant_action_fn(reference_levers(cfg).as_array().astype(np.float32))

    arms: dict[str, dict[str, list[float]]] = {
        "rule_based": evaluate_arm(
            cfg,
            PolicyBook(regulator=regulator, firm=rule_based_firm_fn()),
            seeds,
            model_factory=model_factory,
        ),
        "strategic": evaluate_arm(
            cfg,
            PolicyBook(regulator=regulator, firm=book.firm),
            seeds,
            model_factory=model_factory,
        ),
    }
    if marl_regulator is not None:
        arms["strategic_marl_regulator"] = evaluate_arm(
            cfg,
            PolicyBook(regulator=marl_regulator, firm=book.firm),
            seeds,
            model_factory=model_factory,
        )

    return {
        "schema": SCHEMA,
        "claim": "C6",
        "profile": cfg.profile_name,
        "seed": cfg.seed,
        "degraded": True,
        "backend": BACKEND,
        "note": (
            "RLlib was not used: PLAN.md guardrail 11 makes it non-gating and Stage 10d "
            "sanctions hand-rolled independent PPO with iterated best response. Firms share "
            "one policy (parameter sharing) and the training ego rotates across the strategic "
            "firms. The rule_based arm drives the same environment with zero strategic "
            "actions, which maps exactly onto StrategicControls.neutral."
        ),
        "n_strategic_firms": int(cfg.env.n_strategic_firms),
        "n_eval_episodes": episodes,
        "eval_seeds": seeds,
        "regulator_reference_policy": cfg.policy.name,
        "arms": {
            name: {
                "n": len(next(iter(values.values()))) if values else 0,
                "metrics": {key: _stats_json(_stats(values[key])) for key in sorted(values)},
            }
            for name, values in arms.items()
        },
        "comparison": compare_arms(arms["rule_based"], arms["strategic"]),
    }


# --------------------------------------------------------------------------- #
# stage entry point                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class MarlResult:
    checkpoints: list[Path]
    comparison: Path
    summary: Path
    metrics: dict[str, float] = field(default_factory=dict)


def _flat_metrics(payload: dict[str, Any]) -> dict[str, float]:
    comparison = payload["comparison"]["metrics"]
    metrics: dict[str, float] = {
        "c6_n_eval_episodes": float(payload["n_eval_episodes"]),
        "c6_any_changed": float(payload["comparison"]["any_changed"]),
    }
    for name, block in comparison.items():
        metrics[f"c6_{name}_baseline"] = float(block["baseline"]["mean"])
        metrics[f"c6_{name}_strategic"] = float(block["strategic"]["mean"])
        metrics[f"c6_{name}_diff"] = float(block["diff"])
        metrics[f"c6_{name}_changed"] = float(block["changed"])
    return metrics


def train_marl(
    cfg: RegWorldConfig,
    *,
    model_factory: ModelFactory | None = None,
    n_eval_episodes: int | None = None,
) -> MarlResult:
    """Stage 10d: IPPO by iterated best response, then the C6 comparison."""
    if cfg.rl.marl_backend != "ippo":
        log.warning(
            "cfg.rl.marl_backend=%r requested; this build only ships the sanctioned "
            "IPPO/iterated-best-response fallback (PLAN.md guardrail 11) — running it and "
            "reporting DEGRADED",
            cfg.rl.marl_backend,
        )
    rounds = IBR_ROUNDS.get(cfg.profile_name, 3)
    budget = max(int(cfg.rl.marl_timesteps), 0)
    per_fit = max(MIN_FIT_TIMESTEPS, budget // max(2 * rounds, 1))
    log.info(
        "Stage 10d (DEGRADED, backend=%s): %d best-response round(s) x 2 fits x %d timesteps",
        BACKEND,
        rounds,
        per_fit,
    )

    book = reference_book(cfg)
    firm_agent: PPO | None = None
    regulator_agent: PPO | None = None
    firm_encoder: ObservationEncoder | None = None
    regulator_encoder: ObservationEncoder | None = None
    history: list[dict[str, Any]] = []

    for round_index in range(rounds):
        firm_view = SingleAgentView(
            cfg,
            "firm",
            book,
            model_factory=model_factory,
            seed=cfg.seed + 1000 * round_index + 1,
        )
        try:
            firm_agent, firm_stats = fit_best_response(
                firm_view, per_fit, cfg.seed + 1000 * round_index + 1
            )
            firm_encoder = firm_view.encoder
        finally:
            firm_view.close()
        book = replace(book, firm=sb3_action_fn(firm_agent, firm_encoder))
        history.append({"round": round_index + 1, "agent": "firm", **firm_stats})

        regulator_view = SingleAgentView(
            cfg,
            "regulator",
            book,
            model_factory=model_factory,
            seed=cfg.seed + 1000 * round_index + 2,
        )
        try:
            regulator_agent, regulator_stats = fit_best_response(
                regulator_view, per_fit, cfg.seed + 1000 * round_index + 2
            )
            regulator_encoder = regulator_view.encoder
        finally:
            regulator_view.close()
        book = replace(book, regulator=sb3_action_fn(regulator_agent, regulator_encoder))
        history.append({"round": round_index + 1, "agent": "regulator", **regulator_stats})

    if firm_agent is None or regulator_agent is None or regulator_encoder is None:
        raise RuntimeError("no best-response round ran; check IBR_ROUNDS for this profile")

    out = marl_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    firm_agent.save(str(out / "firm_policy"))
    regulator_agent.save(str(out / "regulator_policy"))
    checkpoints = [out / "firm_policy.zip", out / "regulator_policy.zip"]

    payload = compare_c6(
        cfg,
        book,
        marl_regulator=sb3_action_fn(regulator_agent, regulator_encoder),
        model_factory=model_factory,
        n_episodes=n_eval_episodes,
    )
    payload["training"] = {
        "rounds": rounds,
        "timesteps_per_fit": per_fit,
        "budget_timesteps": budget,
        "total_actual_timesteps": float(sum(entry["actual_timesteps"] for entry in history)),
        "parameter_sharing": True,
        "history": history,
    }
    comparison = comparison_path(cfg)
    comparison.write_text(json.dumps(payload, indent=2))

    metrics = _flat_metrics(payload)
    summary = out / "marl_summary.json"
    summary.write_text(
        json.dumps(
            {
                "profile": cfg.profile_name,
                "backend": BACKEND,
                "degraded": True,
                "training": payload["training"],
                "verdict": payload["comparison"]["verdict"],
                "metrics": metrics,
                "comparison_path": str(comparison),
            },
            indent=2,
        )
    )
    log.info(
        "Stage 10d done: C6 verdict %r (changed: %s) -> %s",
        payload["comparison"]["verdict"],
        payload["comparison"]["changed_metrics"] or "none",
        comparison,
    )
    return MarlResult(
        checkpoints=checkpoints, comparison=comparison, summary=summary, metrics=metrics
    )


__all__ = [
    "BACKEND",
    "C6_HEADLINE_METRICS",
    "C6_METRICS",
    "IBR_ROUNDS",
    "MIN_FIT_TIMESTEPS",
    "REGULATOR",
    "SCHEMA",
    "AgentActionFn",
    "MarlResult",
    "ObservationEncoder",
    "PolicyBook",
    "SingleAgentView",
    "compare_arms",
    "compare_c6",
    "comparison_path",
    "constant_action_fn",
    "episode_seed",
    "evaluate_arm",
    "fit_best_response",
    "marl_dir",
    "reference_book",
    "reference_levers",
    "rollout_arm",
    "rule_based_firm_fn",
    "sb3_action_fn",
    "train_marl",
]
