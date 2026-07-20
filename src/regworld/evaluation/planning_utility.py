"""§11 family 5 — planning utility: does the model make an agent better?

The model-based acid test needs trained policies (SB3 PPO in the emulator, the
TorchRL Dreamer agent, fixed and random baselines) evaluated in the true ABM.
Those artifacts are produced by Stage 10 (Phase 6); this module defines the
comparison and reports honestly that it cannot run yet.
"""

from __future__ import annotations

from pathlib import Path

from regworld.types import RegWorldConfig


def evaluate(cfg: RegWorldConfig) -> dict[str, object]:
    policy_dir = Path(cfg.paths.root) / "rl"
    if not policy_dir.exists():
        return {
            "status": "pending Phase 6 (Stage 10): no trained policies yet",
            "planned_comparison": (
                "J_ABM(pi_emulator) vs pi_ABM-trained vs pi_fixed vs pi_random, "
                "5 seeds x 64 posterior draws; Dreamer exploitation gap "
                "J_emulator - J_ABM <= 15%"
            ),
            "thresholds_dev": {
                "learned_beats_baselines": "non-overlapping 95% CI vs random & fixed",
                "dreamer_exploitation_gap": "<= 15%",
            },
        }
    return {"status": "policies found; full comparison runs in the Stage-10 gate"}
