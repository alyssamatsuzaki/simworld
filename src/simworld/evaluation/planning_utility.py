"""§11 family 5 — planning utility: does the model make an agent better?

The model-based acid test: every available policy (fixed baselines, random,
and any learned artifact Stage 10 has already produced) is rolled out in the
**true ABM** (never the emulator that may have trained it), across a
config-scaled ``seeds x draws`` grid. A learned policy must beat the fixed
and random baselines with a non-overlapping 95% CI. The Dreamer agent is also
rolled out in the emulator so its exploitation gap ``J_emulator - J_ABM`` can
be reported against the <= 15% threshold: a policy that looks brilliant in
the model and mediocre in the real thing has found the model's errors and
steered into them.

Degrades honestly when Stage 10 has not produced any learned policy yet
(reports what is missing) rather than crashing or faking a result.
"""

from __future__ import annotations

import logging
from pathlib import Path

from simworld.agents import planning, registry
from simworld.agents.registry import load_checkpoint_compat
from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)

_THRESHOLDS_DEV = {
    "learned_beats_baselines": "non-overlapping 95% CI vs random & fixed baselines",
    "dreamer_exploitation_gap": "(J_emulator - J_ABM) / |J_ABM| <= 0.15",
}
_LEARNED_NAMES = ("rl_ppo", "rl_dreamer")
_BASELINE_NAMES = ("random", "none")


def _scaled_seeds_and_draws(cfg: SimWorldConfig) -> tuple[list[int], int]:
    """Bound the Monte-Carlo grid so the smoke profile stays fast (§10 gate)."""
    n_seeds = 2 if cfg.profile_name == "smoke" else min(len(cfg.seeds), 5)
    seeds = list(cfg.seeds[:n_seeds]) or [cfg.seed]
    draws = max(1, min(cfg.eval.abm_validation_episodes // max(n_seeds, 1), 4))
    return seeds, draws


def evaluate(cfg: SimWorldConfig) -> dict[str, object]:
    policy_dir = Path(cfg.paths.root) / "rl"
    names = registry.available_policies(cfg)
    learned = [name for name in _LEARNED_NAMES if name in names]
    missing = [name for name in _LEARNED_NAMES if name not in names]

    if not policy_dir.exists() or not learned:
        return {
            "status": "no trained policies found; run `make rl` (Stage 10) first",
            "available": names,
            "missing": missing,
            "thresholds_dev": _THRESHOLDS_DEV,
        }

    seeds, draws = _scaled_seeds_and_draws(cfg)
    per_policy: dict[str, dict[str, object]] = {}
    for name in names:
        try:
            policy_fn = registry.load_policy(cfg, name)
        except Exception:
            log.exception("policy %s failed to load; skipping", name)
            continue
        stats = planning.evaluate_in_abm(cfg, policy_fn, seeds, draws=draws)
        lo, hi = stats.ci95()
        per_policy[name] = {
            "mean_return": stats.mean,
            "std_return": stats.std,
            "n": stats.n,
            "ci95": [lo, hi],
        }

    baseline_names = [name for name in _BASELINE_NAMES if name in per_policy]
    verdicts: dict[str, bool] = {}
    for learned_name in learned:
        entry = per_policy.get(learned_name)
        if entry is None:
            continue
        learned_lo = float(entry["ci95"][0])  # type: ignore[index]
        beats_all = True
        for baseline in baseline_names:
            baseline_hi = float(per_policy[baseline]["ci95"][1])  # type: ignore[index]
            beats_all = beats_all and (learned_lo > baseline_hi)
        verdicts[learned_name] = beats_all

    result: dict[str, object] = {
        "seeds": seeds,
        "draws_per_seed": draws,
        "policies": per_policy,
        "learned_beats_baselines": verdicts,
        "missing": missing,
        "thresholds_dev": _THRESHOLDS_DEV,
    }

    if "rl_dreamer" in learned and "rl_dreamer" in per_policy:
        try:
            model, meta = load_checkpoint_compat(cfg)
            dreamer_fn = registry.load_policy(cfg, "rl_dreamer")
            j_abm = float(per_policy["rl_dreamer"]["mean_return"])  # type: ignore[arg-type]
            emulator_stats = planning.evaluate_in_emulator(
                cfg, dreamer_fn, model, meta, seeds, draws=draws
            )
            gap = planning.exploitation_gap(emulator_stats.mean, j_abm)
            result["dreamer_exploitation_gap"] = {
                "j_emulator": emulator_stats.mean,
                "j_abm": j_abm,
                "gap": gap,
                "within_threshold": bool(gap <= 0.15),
            }
        except Exception:
            log.exception("dreamer exploitation-gap check failed")
            result["dreamer_exploitation_gap"] = {"status": "failed; see log"}

    return result
