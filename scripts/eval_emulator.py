"""§11 driver: run every metric family, write reports/eval/{report.md, metrics.json, figures/}.

Families that cannot run yet (planning utility before Phase 6, sensitivity
before Stage 14) report their status honestly. Any family that CRASHES is
recorded with its traceback and the script exits nonzero — a broken grader
must not look like a passing one.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = get_logger(__name__)


def _figures(metrics: dict[str, object], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    predictive = metrics.get("predictive")
    if isinstance(predictive, dict) and "compliance_drift" in predictive:
        drift = predictive["compliance_drift"]
        persistence = predictive["persistence_drift"]
        ks = [int(k[1:]) for k in drift]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, [drift[f"k{k}"] for k in ks], "o-", label="GraphRSSM")
        ax.plot(ks, [persistence[f"k{k}"] for k in ks], "s--", label="persistence")
        ax.axhline(0.10, color="grey", lw=0.8, ls=":", label="useful-range bound")
        ax.set_xlabel("imagination horizon k (quarters)")
        ax.set_ylabel("compliance MAE")
        ax.set_title("k-step open-loop drift (held-out episodes)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_kstep_drift.png", dpi=150)
        plt.close(fig)
        written.append("fig_kstep_drift.png")

    calibration = metrics.get("calibration")
    if isinstance(calibration, dict) and "reliability_diagram" in calibration:
        bins = [b for b in calibration["reliability_diagram"] if b.get("confidence") is not None]
        if bins:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                "o-",
                label=f"ECE {calibration['ece']:.3f}",
            )
            ax.set_xlabel("predicted compliance probability")
            ax.set_ylabel("empirical frequency")
            ax.set_title("one-step node reliability")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_reliability.png", dpi=150)
            plt.close(fig)
            written.append("fig_reliability.png")

    distributional = metrics.get("distributional")
    if isinstance(distributional, dict) and "abm_terminal_compliance" in distributional:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            distributional["abm_terminal_compliance"], bins=12, alpha=0.6, label="ABM", density=True
        )
        ax.hist(
            distributional["emulator_terminal_compliance"],
            bins=12,
            alpha=0.6,
            label="emulator",
            density=True,
        )
        ax.set_xlabel("terminal compliance rate")
        ax.set_title(
            f"terminal distributions (W1 {distributional['w1_compliance']:.3f}, "
            f"perm p {distributional['permutation_p_compliance']:.2f})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_terminal_dist.png", dpi=150)
        plt.close(fig)
        written.append("fig_terminal_dist.png")

    ood = metrics.get("ood")
    if isinstance(ood, dict) and "enforcement_1p5_error" in ood:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(
            ["in-range\n(held-out)", "enforcement 1.5\n(1.5x outside range)"],
            [ood["heldout_mean_error"], ood["enforcement_1p5_error"]],
            color=["tab:blue", "tab:red"],
        )
        ax.set_ylabel("compliance MAE")
        ax.set_title("Fig 13: error growth out of distribution")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_ood.png", dpi=150)
        plt.close(fig)
        written.append("fig_ood.png")
    return written


def _report(metrics: dict[str, object], errors: dict[str, str], out: Path) -> None:
    lines = [
        "# §11 evaluation report",
        "",
        f"Profile: `{metrics['profile']}` — thresholds are pass criteria at `dev`; "
        "at smoke every number is reported against its threshold without gating.",
        "",
    ]
    order = [
        ("predictive", "1 · Predictive accuracy"),
        ("distributional", "2 · Distributional fidelity"),
        ("calibration", "3 · Calibration"),
        ("dtw", "4 · Trajectory shape (DTW)"),
        ("planning_utility", "5 · Planning utility"),
        ("behavioral_fidelity", "6 · Behavioral fidelity"),
        ("parameter_recovery", "7 · Parameter recovery (C1)"),
        ("causal_eval", "8 · Causal evaluation (C2/5f)"),
        ("ood", "9 · Out-of-distribution"),
        ("backtest", "10 · Historical backtest"),
        ("ablations", "11 · Ablations"),
        ("sensitivity", "12 · Sensitivity"),
    ]
    for key, title in order:
        lines.append(f"## {title}")
        lines.append("")
        if key in errors:
            lines.append(f"**FAILED**: `{errors[key].splitlines()[-1]}` (traceback in log)")
        else:
            payload = metrics.get(key, {"status": "not run"})
            lines.append("```json")
            lines.append(json.dumps(payload, indent=2, default=str))
            lines.append("```")
        lines.append("")
    out.write_text("\n".join(lines))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    from regworld.evaluation import (
        ablations,
        backtest,
        behavioral_fidelity,
        calibration_curves,
        causal_eval,
        distributional,
        dtw,
        harness,
        ood,
        parameter_recovery,
        planning_utility,
        predictive,
    )

    ctx = harness.load_context(cfg_obj)
    metrics: dict[str, object] = {"profile": cfg_obj.profile_name, "seed": cfg_obj.seed}
    errors: dict[str, str] = {}

    families: list[tuple[str, Callable[[], object]]] = [
        ("predictive", lambda: predictive.evaluate(ctx)),
        ("distributional", lambda: distributional.evaluate(ctx)),
        ("calibration", lambda: calibration_curves.evaluate(ctx)),
        ("dtw", lambda: dtw.evaluate(ctx)),
        ("planning_utility", lambda: planning_utility.evaluate(cfg_obj)),
        ("behavioral_fidelity", lambda: behavioral_fidelity.evaluate(ctx)),
        ("parameter_recovery", lambda: parameter_recovery.evaluate(cfg_obj)),
        ("causal_eval", lambda: causal_eval.evaluate(cfg_obj)),
        ("ood", lambda: ood.evaluate(ctx)),
        ("backtest", lambda: backtest.evaluate(ctx)),
        ("ablations", lambda: ablations.evaluate(cfg_obj)),
        (
            "sensitivity",
            lambda: {"status": "Stage 14 (Phase 6): Morris/Sobol run on this emulator"},
        ),
    ]
    for name, run in families:
        try:
            metrics[name] = run()
            log.info("family %s done", name)
        except Exception:
            errors[name] = traceback.format_exc()
            log.exception("family %s FAILED", name)

    eval_dir = Path(cfg_obj.paths.reports) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics["figures"] = _figures(metrics, eval_dir / "figures")
    metrics["errors"] = {k: v.splitlines()[-1] for k, v in errors.items()}
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _report(metrics, errors, eval_dir / "report.md")
    log.info("evaluation report -> %s", eval_dir / "report.md")
    if errors:
        log.error("%d metric families failed: %s", len(errors), sorted(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
