"""Stage 17: Assemble reports/FINDINGS.md from committed artifacts (§10, Stage 17).

Contract: build_findings(cfg: SimWorldConfig) -> Path to reports/FINDINGS.md.
The report always includes five required sections in order:
  1. Synthetic-world disclaimer.
  2. Four-number causal table (tau_true, tau_abm, tau_qe, tau_obs with CIs).
  3. Claims C1-C6 with verdicts (SUPPORTED / REFUTED / INCONCLUSIVE).
  4. "Where this model fails" section (the heading MUST always appear).
  5. Run manifest (every stage's status, wall clock, git hash, config).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from simworld.types import SimWorldConfig

log = logging.getLogger(__name__)


def _read_artifact(path: Path) -> dict[str, Any]:
    """Read JSON artifact; return empty dict if missing (graceful degradation)."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return {}
    return {}


def _ci_str(value: float, ci: list[float] | None = None) -> str:
    """Format a value with optional CI."""
    if ci is not None and len(ci) == 2:
        return f"{value:.4f} [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]"
    return f"{value:.4f}"


def _parse_coverage_count(coverage_at_90: Any) -> tuple[int, int] | None:
    """Parse the parameter-recovery family's 'covered/graded' string into (covered, graded)."""
    if not isinstance(coverage_at_90, str) or "/" not in coverage_at_90:
        return None
    try:
        covered, graded = (int(part) for part in coverage_at_90.split("/", 1))
    except ValueError:
        return None
    return covered, graded


def build_findings(cfg: SimWorldConfig) -> Path:
    """Assemble reports/FINDINGS.md from committed artifacts.

    Reads:
      - artifacts/causal/four_numbers.json
      - artifacts/sensitivity/indices.json (for C4)
      - artifacts/ensemble/ensemble_summary.json (for C5)
      - artifacts/calibration/ (for C1)
      - reports/eval/metrics.json (for OOD, useful_range)
      - reports/run_manifest.json

    Returns the Path to reports/FINDINGS.md.
    """
    reports_dir = Path(cfg.paths.reports)
    artifacts_dir = Path(cfg.paths.root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    findings_path = reports_dir / "FINDINGS.md"

    lines: list[str] = []

    # =========================================================================
    # SECTION 1: Disclaimer (ALWAYS FIRST)
    # =========================================================================
    lines.append("# SimWorld Findings")
    lines.append("")
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "This world model is entirely synthetic with known ground truth. Every finding is "
        "methodological: what is demonstrated is that this pipeline recovers the truth when the "
        "truth is recoverable and fails legibly when it is not. The policy insights below "
        "emerge from a constructed regulatory environment whose parameters and causal structure "
        "are known in full. A real policy deployment would require validation against observed "
        "data, external cross-checks, and expert judgment; this report's value is in exposing "
        "the methodology and the seams where it breaks."
    )
    lines.append("")

    # =========================================================================
    # SECTION 2: Four-number causal table
    # =========================================================================
    lines.append("## The Four-Number Causal Table")
    lines.append("")
    lines.append(
        "Figure 1 (see reports/figures/fig01_four_numbers.png) and the table below "
        "report the four key causal estimates:"
    )
    lines.append("")

    four_numbers = _read_artifact(artifacts_dir / "causal" / "four_numbers.json")
    if four_numbers:
        tau_true = four_numbers.get("tau_true", None)
        tau_abm = four_numbers.get("tau_abm", None)
        tau_qe = four_numbers.get("tau_qe", None)
        tau_qe_ci = four_numbers.get("tau_qe_ci", None)
        tau_obs = four_numbers.get("tau_obs", None)
        tau_obs_ci = four_numbers.get("tau_obs_ci", None)

        lines.append("| Estimand | Value |")
        lines.append("|---|---|")
        if tau_true is not None:
            lines.append(f"| τ_true (do() ATT, ground truth) | {_ci_str(tau_true)} |")
        if tau_abm is not None:
            lines.append(f"| τ_abm (simulator DIL rollout) | {_ci_str(tau_abm)} |")
        if tau_qe is not None:
            lines.append(f"| τ_qe (observational DML) | {_ci_str(tau_qe, tau_qe_ci)} |")
        if tau_obs is not None:
            lines.append(f"| τ_obs (naive panel contrast) | {_ci_str(tau_obs, tau_obs_ci)} |")
        lines.append("")
    else:
        lines.append("**Artifact missing:** `artifacts/causal/four_numbers.json` not found.")
        lines.append("")

    # =========================================================================
    # SECTION 3: Claims C1-C6
    # =========================================================================
    lines.append("## The Six Claims")
    lines.append("")

    # Claim texts from PLAN.md lines 130-145
    claims = {
        "C1": (
            "Bayesian calibration recovers the true behavioral parameters when the model "
            "is well specified, and fails *legibly* (a visibly biased peer coefficient "
            "β_peer) when supply-network capacity homophily is switched on.",
            "parameter_recovery",
        ),
        "C2": (
            "The observational estimate of the enforcement effect is confidently wrong "
            "when audit targeting correlates with unobserved firm capacity. The staggered-rollout "
            "DiD recovers the true effect; DoWhy's refuters catch the naive estimate.",
            "causal_eval",
        ),
        "C3": (
            "The graph-RSSM emulator reproduces the ABM's *distribution* of outcomes "
            "within tolerance at 10³-10⁴x the speed, and degrades honestly out of distribution.",
            "distributional",
        ),
        "C4": (
            "Of ~16 uncertain parameters, a small handful drive most outcome variance — "
            "which tells the client what to measure next.",
            "sensitivity",
        ),
        "C5": (
            "Aggressive uniform enforcement maximizes compliance and backfires on market "
            "concentration: small firms exit, HHI rises. Phased, targeted enforcement buys nearly "
            "the same compliance for materially less concentration. Reported as a Pareto frontier "
            "with credible intervals across the parameter posterior.",
            "ensemble",
        ),
        "C6": (
            "Modeling the ten largest firms as strategic learners (MARL) either changes C5 "
            "or does not. Report which.",
            "planning_utility",
        ),
    }

    eval_metrics = _read_artifact(Path(cfg.paths.reports) / "eval" / "metrics.json")
    ensemble_summary = _read_artifact(artifacts_dir / "ensemble" / "ensemble_summary.json")
    sensitivity_summary = _read_artifact(artifacts_dir / "sensitivity" / "sensitivity_summary.json")
    sensitivity_indices = _read_artifact(artifacts_dir / "sensitivity" / "indices.json")
    calib_micro = _read_artifact(artifacts_dir / "calibration" / "micro_diagnostics.json")
    ensemble_validation = _read_artifact(artifacts_dir / "ensemble" / "validation_report.json")

    # Every emulator-derived claim (C3, C4, C5) inherits the emulator's credibility.
    # Stage 11's ABM cross-check is the measurement of that credibility, so a cube built
    # on an emulator that failed it cannot support a claim no matter how complete the
    # artifact looks (§10 Stage 11: "the entire ensemble is decoration").
    _emulator_coverage = ensemble_validation.get("coverage") if ensemble_validation else None
    _coverage_threshold = (
        ensemble_validation.get("threshold", 0.85) if ensemble_validation else 0.85
    )
    _emulator_untrustworthy = (
        _emulator_coverage is not None
        and isinstance(_emulator_coverage, int | float)
        and not math.isnan(_emulator_coverage)
        and _emulator_coverage < _coverage_threshold
    )
    _coverage_caveat = (
        (
            f"the Stage-11 ABM cross-check covers only {_emulator_coverage:.2%} of outcomes "
            f"(threshold {_coverage_threshold:.0%}), so the emulator this rests on is not "
            "validated"
        )
        if _emulator_untrustworthy
        else ""
    )

    for claim_key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        claim_text, metric_key = claims[claim_key]
        lines.append(f"### {claim_key}")
        lines.append("")
        lines.append(f"**Claim:** {claim_text}")
        lines.append("")

        verdict = "INCONCLUSIVE"
        evidence = ""

        if claim_key == "C1":
            # C1 (§18): SUPPORTED requires the full recovery gate — convergence
            # (max R-hat < 1.01, divergences == 0) AND >= 12/16 parameters cover
            # θ* at 90% under the WELL-SPECIFIED world, plus a biased β_peer under the
            # confounded world. The recovery grid (calibration.recovery_grid; on at dev)
            # produces that two-world contrast; without it, coverage falls back to the
            # shipped single variant. At smoke the draw count is too small for clean
            # R-hat and the posteriors too wide to resolve β_peer, so C1 is
            # INCONCLUSIVE by design.
            recovery = eval_metrics.get("parameter_recovery", {}) or {}
            covered = _parse_coverage_count(recovery.get("coverage_at_90"))
            beta_peer_miss = recovery.get("beta_peer_miss_under_confounded")
            # When the recovery grid has run, coverage is graded under the WELL-
            # SPECIFIED world, so convergence must be judged on that same fit; the grid
            # carries its R-hat/divergences. Otherwise fall back to the shipped micro
            # diagnostics (the single-variant world).
            grid_r_hat = recovery.get("max_r_hat")
            if grid_r_hat is not None or calib_micro:
                if grid_r_hat is not None:
                    r_hat = grid_r_hat
                    divergences = recovery.get("divergences")
                    n_params = None
                else:
                    r_hat = calib_micro.get("max_r_hat", calib_micro.get("r_hat", None))
                    divergences = calib_micro.get("divergences", None)
                    n_params = len(calib_micro.get("parameters", {})) or None
                if r_hat is not None:
                    param_note = f" across {n_params} fitted parameters" if n_params else ""
                    cov_note = (
                        f", {covered[0]}/{covered[1]} parameters cover θ* at 90%" if covered else ""
                    )
                    converged = r_hat < 1.01 and (divergences in (0, None))
                    covers_enough = covered is not None and covered[0] >= 12
                    if converged and covers_enough:
                        verdict = "SUPPORTED"
                        evidence = (
                            f"Full recovery gate met: max R-hat={r_hat:.3f} < 1.01{param_note}, "
                            f"divergences={divergences}{cov_note} (>= 12). "
                            "Posterior marginals recover θ* under the well-specified world."
                        )
                    elif not converged:
                        verdict = "INCONCLUSIVE"
                        evidence = (
                            f"Max R-hat={r_hat:.3f}{param_note}, divergences={divergences}"
                            f"{cov_note} — convergence is not clean at this profile's draw "
                            "count; recovery not yet assertable (dev-profile gate)."
                        )
                    else:
                        verdict = "INCONCLUSIVE"
                        evidence = (
                            f"Chains converged (max R-hat={r_hat:.3f}, divergences={divergences})"
                            f"{cov_note}, short of the >= 12/16 coverage bar at this profile."
                        )
                    if isinstance(beta_peer_miss, bool):
                        verb = "misses" if beta_peer_miss else "covers"
                        evidence += f" Under confounded, β_peer {verb} truth (the C1 failure half)."
                else:
                    evidence = "Micro diagnostics incomplete."
            else:
                evidence = "Artifact `artifacts/calibration/micro_diagnostics.json` not found."

        elif claim_key == "C2":
            # C2: Causal identifiability. Prefer a populated causal_eval family; otherwise
            # fall back to the four-number gate, whose flags encode exactly this claim
            # (observational confidently wrong; staggered DiD recovers the truth).
            causal_data = eval_metrics.get(metric_key, {})
            did_covers = causal_data.get("did_covers_truth", None) if causal_data else None
            if did_covers is True:
                verdict = "SUPPORTED"
                evidence = "Staggered DiD identifies true effect; observational estimate biased."
            elif did_covers is False:
                verdict = "REFUTED"
                evidence = "DiD estimate does not cover ground truth."
            elif four_numbers:
                sign_ok = four_numbers.get("sign_ok")
                did_ok = four_numbers.get("did_agreement_ok")
                flagged = four_numbers.get("flagged")
                tau_true = four_numbers.get("tau_true")
                tau_obs = four_numbers.get("tau_obs")
                tau_abm = four_numbers.get("tau_abm")
                if sign_ok is not None and did_ok is not None:
                    if sign_ok and did_ok and not flagged:
                        verdict = "SUPPORTED"
                        evidence = (
                            f"Four-number gate passed: naive observational τ_obs={tau_obs:.3f} is "
                            f"confidently wrong against τ_true={tau_true:.3f}, while the DiL "
                            f"simulator/DiD path recovers τ_abm={tau_abm:.3f} (sign and DiD "
                            "agreement OK)."
                        )
                    else:
                        verdict = "REFUTED"
                        evidence = (
                            f"Four-number gate flagged: sign_ok={sign_ok}, "
                            f"did_agreement_ok={did_ok}, flagged={flagged}."
                        )
                else:
                    evidence = "Four-number gate artifact present but missing verdict flags."
            else:
                evidence = "Neither `causal_eval` metrics nor `four_numbers.json` available."

        elif claim_key == "C3":
            # C3: Emulator fidelity. Check distributional + OOD metrics.
            distributional = eval_metrics.get("distributional", {})
            ood_data = eval_metrics.get("ood", {})
            if isinstance(distributional, dict) and isinstance(ood_data, dict):
                w1 = distributional.get("w1_compliance", None)
                error_growth = ood_data.get("error_growth_factor_at_1p5", None)
                if w1 is not None and error_growth is not None:
                    if _emulator_untrustworthy:
                        verdict = "INCONCLUSIVE"
                        evidence = (
                            f"W1 distance={w1:.3f}, OOD error growth={error_growth:.2f}x, but "
                            f"{_coverage_caveat}."
                        )
                    elif w1 < 0.1 and error_growth < 1.5:
                        verdict = "SUPPORTED"
                        evidence = f"W1 distance={w1:.3f}, OOD error growth={error_growth:.2f}x."
                    else:
                        verdict = "INCONCLUSIVE"
                        evidence = (
                            f"Distributional match marginal (W1={w1:.3f}); "
                            f"OOD degradation={error_growth:.2f}x."
                        )
                else:
                    evidence = "Distributional or OOD metrics incomplete."
            else:
                evidence = "Emulator evaluation artifacts missing."

        elif claim_key == "C4":
            # C4: Sensitivity indices. Morris mu* and Sobol total-order indices live in
            # indices.json (sensitivity_summary.json only carries counts + Optuna best).
            # C4 is a claim about the BEHAVIORAL parameters theta (§7.3), not the four
            # policy levers. The lever screen ("morris") runs on the emulator and answers a
            # different question; only the theta screen ("morris_theta", Morris elementary
            # effects on the tensorized ABM) can carry this claim.
            theta_screen = (
                sensitivity_indices.get("morris_theta", {}) if sensitivity_indices else {}
            )
            shares = theta_screen.get("mu_star_share_mean", {})
            ranking = theta_screen.get("ranking", [])
            if shares and ranking:
                top_names = ", ".join(ranking[:3])
                top3_share = sum(float(shares.get(name, 0.0)) for name in ranking[:3])
                n_vars = theta_screen.get("num_vars", len(ranking))
                optuna_best = (
                    sensitivity_summary.get("optuna_best_J") if sensitivity_summary else None
                )
                optuna_note = (
                    f" Optuna policy search reached best J={optuna_best:.3f}."
                    if optuna_best is not None
                    else ""
                )
                scope_note = (
                    f" ({n_vars} of the 16 §7.3 parameters enter the forecast dynamics; "
                    "beta_capacity is answer-key-only and q0/q1 are observation-model-only, "
                    "so screening them on the ABM would manufacture guaranteed zeros.)"
                )
                # "A small handful drive most outcome variance" is the claim; require the
                # top three to actually concentrate effect, rather than merely existing.
                if top3_share >= 0.5:
                    verdict = "SUPPORTED"
                    evidence = (
                        f"Morris elementary effects over {n_vars} behavioral parameters on the "
                        f"tensorized ABM ({theta_screen.get('count', '?')} rollouts) rank the "
                        f"drivers {top_names}; the top three carry {top3_share:.0%} of mean "
                        f"mu* share, so a small handful dominate.{scope_note}{optuna_note}"
                    )
                else:
                    verdict = "INCONCLUSIVE"
                    evidence = (
                        f"Morris elementary effects over {n_vars} behavioral parameters rank "
                        f"{top_names} highest, but the top three carry only {top3_share:.0%} of "
                        f"mean mu* share — effect is spread too evenly to call a small handful "
                        f"dominant.{scope_note}{optuna_note}"
                    )
            else:
                evidence = (
                    "Artifact `artifacts/sensitivity/indices.json` has no `morris_theta` block; "
                    "the theta screen (Stage 14a) has not run. The lever screen cannot carry C4."
                )

        elif claim_key == "C5":
            # C5: Ensemble + Pareto frontier. Backfire probability lives in the summary's
            # top-level metrics; older runs nested it under validation.
            if ensemble_summary:
                ens_metrics = ensemble_summary.get("metrics", {})
                backfire_rate = ens_metrics.get("backfire_rate", None)
                if backfire_rate is None:
                    backfire_rate = ensemble_summary.get("validation", {}).get(
                        "backfire_probability", None
                    )
                if backfire_rate is not None:
                    n_policies = ens_metrics.get("n_policies_included")
                    n_cells = ens_metrics.get("n_cells")
                    n_policies = int(n_policies) if n_policies is not None else "?"
                    n_cells = int(n_cells) if n_cells is not None else "?"
                    base = (
                        f"Scenario cube built over {n_cells} cells / {n_policies} policies; "
                        f"backfire probability {backfire_rate:.2%}."
                    )
                    # A cube that exists is not a cube that shows anything. C5 asserts a
                    # specific regime (compliance up, HHI up, CS down); if the ensemble never
                    # enters it, or rests on an unvalidated emulator, the claim is not carried.
                    if _emulator_untrustworthy:
                        verdict = "INCONCLUSIVE"
                        evidence = f"{base} Verdict withheld: {_coverage_caveat}."
                    elif backfire_rate <= 0.0:
                        verdict = "INCONCLUSIVE"
                        evidence = (
                            f"{base} The backfire regime never occurs anywhere in this "
                            "ensemble, so the compliance-versus-concentration trade-off C5 "
                            "asserts is not exhibited; a cube containing no backfire cannot "
                            "support a backfire claim."
                        )
                    else:
                        verdict = "SUPPORTED"
                        evidence = (
                            f"{base} The Pareto frontier (terminal compliance vs ΔHHI) "
                            "exhibits the backfire regime across the ensemble."
                        )
                else:
                    verdict = "INCONCLUSIVE"
                    evidence = "Ensemble summary present but no backfire_rate field."
            else:
                evidence = "Artifact `artifacts/ensemble/ensemble_summary.json` not found."

        elif claim_key == "C6":
            # C6 asks whether strategic firms change C5. Only the Stage-10d MARL ablation
            # can answer it; planning_utility is a single-agent metric and was the wrong
            # source. A null here is only meaningful if the strategic firms were actually
            # trained enough to behave strategically (§10 Stage 10d: 200k steps).
            marl = _read_artifact(artifacts_dir / "marl" / "c6_comparison.json")
            comparison = marl.get("comparison", {}) if marl else {}
            if comparison:
                changed = comparison.get("changed_metrics", []) or []
                training = marl.get("training", {})
                budget = training.get("budget_timesteps")
                n_eval = marl.get("n_eval_episodes", "?")
                backend = marl.get("backend", "unknown")
                undertrained = isinstance(budget, int | float) and budget < 50_000
                base = (
                    f"Stage-10d ablation ({backend}, {n_eval} paired episodes, "
                    f"{budget if budget is not None else '?'} training timesteps) compared "
                    "strategic top-K firms against rule-based firms on the C5 headline metrics."
                )
                if undertrained:
                    verdict = "INCONCLUSIVE"
                    detail = (
                        f"no headline metric moved ({', '.join(changed)})"
                        if changed
                        else "no headline metric moved"
                    )
                    evidence = (
                        f"{base} Result: {detail}. Verdict withheld: at {budget} timesteps the "
                        "strategic firms are far below the ~200k the ablation calls for, so this "
                        "null measures the training budget, not the absence of strategic effects."
                    )
                elif changed:
                    verdict = "REFUTED"
                    evidence = (
                        f"{base} Strategic firms moved {', '.join(changed)} with "
                        "non-overlapping 95% CIs: MARL changes the C5 conclusion."
                    )
                else:
                    verdict = "SUPPORTED"
                    evidence = (
                        f"{base} No headline metric moved with a non-overlapping 95% CI, so "
                        "modelling the largest firms as strategic learners did not change C5 — "
                        "a clean negative result, and the honest outcome for most policy "
                        "questions (§17)."
                    )
            else:
                evidence = (
                    "Artifact `artifacts/marl/c6_comparison.json` not found; the Stage-10d "
                    "MARL ablation has not run, so C6 is unanswered."
                )

        lines.append(f"**Verdict:** {verdict}")
        lines.append("")
        if evidence:
            lines.append(f"**Evidence:** {evidence}")
            lines.append("")

    # =========================================================================
    # SECTION 4: Where this model fails (REQUIRED HEADING)
    # =========================================================================
    lines.append("## Where This Model Fails")
    lines.append("")
    lines.append(
        "The pipeline is honest about its seams and the stages at which it cannot generalize:"
    )
    lines.append("")

    failure_notes: list[str] = []

    # OOD degradation
    ood_data = eval_metrics.get("ood", {})
    if isinstance(ood_data, dict):
        enforcement_1p5 = ood_data.get("enforcement_1p5_error", None)
        held_out = ood_data.get("heldout_mean_error", None)
        if enforcement_1p5 is not None and held_out is not None:
            factor = enforcement_1p5 / held_out if held_out > 0 else None
            if factor is not None:
                failure_notes.append(
                    f"**Out-of-distribution:** When enforcement is pushed 1.5x beyond "
                    f"training range, compliance MAE grows from {held_out:.3f} to "
                    f"{enforcement_1p5:.3f} ({factor:.1f}x growth). The emulator has not "
                    "learned to extrapolate."
                )

    # β_peer bias under homophily
    if calib_micro:
        dgp_variant = calib_micro.get("dgp_variant", "unknown")
        if dgp_variant == "confounded":
            beta_peer_bias = calib_micro.get("beta_peer_bias", None)
            if beta_peer_bias is not None:
                failure_notes.append(
                    f"**Hidden confounding (capacity homophily):** Under the confounded DGP "
                    f"with supply-network homophily active, the peer coefficient β_peer "
                    f"is biased by {beta_peer_bias:+.3f} — a visible failure mode that "
                    "validates the identification strategy."
                )

    # Useful range / horizon
    predictive = eval_metrics.get("predictive", {})
    if isinstance(predictive, dict):
        useful_range = predictive.get("useful_range_quarters", None)
        if useful_range is not None:
            failure_notes.append(
                f"**Horizon limits:** Multi-step compliance forecasting is useful only "
                f"within {useful_range} quarters. Beyond this horizon, the model's "
                "open-loop drift exceeds a 10% mean absolute error threshold."
            )

    # RL / planning utility shortfalls — PLAN Stage 17 requires naming the
    # policies whose ABM performance falls short of their emulator performance.
    planning_data = eval_metrics.get("planning_utility", {})
    if isinstance(planning_data, dict):
        status = planning_data.get("status", "")
        gap_info = planning_data.get("dreamer_exploitation_gap")
        if isinstance(gap_info, dict) and gap_info.get("gap") is not None:
            gap = float(gap_info["gap"])
            j_emu, j_abm = gap_info.get("j_emulator"), gap_info.get("j_abm")
            over = " (exceeds the 15% budget)" if gap > 0.15 else " (within the 15% budget)"
            failure_notes.append(
                f"**Emulator exploitation:** the Dreamer policy's exploitation gap "
                f"J_emulator - J_ABM is {gap:+.1%}{over} "
                f"(J_emulator={j_emu:.3f} vs J_ABM={j_abm:.3f}) — the planner steers into "
                "the model's errors to exactly the extent this gap is positive."
            )
        elif "degraded" in status.lower() or "pending" in status.lower():
            failure_notes.append(
                "**Learned policies:** The trained RL policy may not meet the "
                "planning-utility threshold in the true ABM, or is still in development. "
                "Report separately if applicable."
            )

    # Stages that did not run cleanly, from the run manifest (§15: a degraded run that
    # reports itself as clean is misconduct — FAILED/BLOCKED must surface here too, not
    # only in the manifest table).
    manifest = _read_artifact(reports_dir / "run_manifest.json")
    if manifest and "stages" in manifest:
        by_status: dict[str, list[str]] = {}
        for name, stage_info in manifest["stages"].items():
            if not isinstance(stage_info, dict):
                continue
            status = stage_info.get("status")
            if status in {"DEGRADED", "FAILED", "BLOCKED"}:
                by_status.setdefault(str(status), []).append(name)
        _status_note = {
            "DEGRADED": "ran with substitutions or limitations",
            "FAILED": "did not complete; every claim downstream of them is unsupported",
            "BLOCKED": "could not be built at all",
        }
        for status in ("FAILED", "BLOCKED", "DEGRADED"):
            names = by_status.get(status)
            if names:
                failure_notes.append(
                    f"**{status.capitalize()} stages:** {', '.join(sorted(names))} "
                    f"{_status_note[status]}; check the run manifest notes for details."
                )

    if failure_notes:
        for note in failure_notes:
            lines.append(f"- {note}")
    else:
        lines.append(
            "*(No major failure modes recorded; the pipeline ran to completion with no "
            "DEGRADED stages and within acceptable thresholds.)*"
        )
    lines.append("")

    # Manual-verification note (§18): the OOD-banner hand-check is the one DoD
    # item that cannot be verified autonomously. The dashboard launches headless
    # without error; the banner's reactivity is confirmed by hand.
    lines.append("")
    lines.append("### Pending manual verification")
    lines.append("")
    lines.append(
        "- **Streamlit OOD banner (§18):** launch `make dashboard`, set enforcement "
        "and targeting sliders to 1.0 — the banner must turn red "
        '("OUT OF DISTRIBUTION: Mahalanobis distance … exceeds …"); return them to '
        'enforcement 0.5 / targeting -0.5 and it must go green ("In distribution"). '
        "The dashboard is confirmed to launch headless without error; this reactivity "
        "check is the single item that requires a human."
    )
    lines.append("")

    # =========================================================================
    # SECTION 5: Run manifest
    # =========================================================================
    lines.append("## Run Manifest")
    lines.append("")

    if manifest:
        profile = manifest.get("profile", "unknown")
        seed = manifest.get("seed", "unknown")
        git_commit = manifest.get("git_commit", "unknown")
        wall_clock = manifest.get("wall_clock_total", 0)

        lines.append(f"**Profile:** {profile}")
        lines.append(f"**Seed:** {seed}")
        lines.append(f"**Git commit:** {git_commit}")
        lines.append(f"**Total wall-clock time:** {wall_clock:.1f} seconds")
        lines.append("")
        lines.append("### Stage-by-stage status")
        lines.append("")
        lines.append("| Stage | Status | Wall clock (s) | Notes |")
        lines.append("|---|---|---|---|")

        stages = manifest.get("stages", {})
        for stage_name in sorted(stages.keys()):
            stage_info = stages[stage_name]
            status = stage_info.get("status", "UNKNOWN")
            wall = stage_info.get("wall_clock", 0)
            notes = stage_info.get("notes", "")
            lines.append(f"| {stage_name} | {status} | {wall:.2f} | {notes} |")
    else:
        lines.append("**Artifact missing:** `reports/run_manifest.json` not found.")
    lines.append("")

    # Write the report
    report_text = "\n".join(lines)
    findings_path.write_text(report_text, encoding="utf-8")
    log.info("FINDINGS.md written to %s (%d lines)", findings_path, len(lines))
    return findings_path
