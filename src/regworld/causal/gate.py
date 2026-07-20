"""THE GATE (5f): four numbers for the effect of enacting the regulation.

| tau_true | do(rollout) vs do(never) in the DGP, sealed | the answer key |
| tau_abm  | the same intervention, calibrated simulator | what our model believes |
| tau_qe   | staggered DiD on the historical panel       | what the data credibly says |
| tau_obs  | DML ignoring the rollout timing             | the careless analyst's number |

Two comparisons, each like-with-like (DEVIATIONS 2026-07-20: the DGP has
cross-region interference, so the DiD's estimand is the sealed ``tau_did_truth``,
not ``tau_true``):

- identification: ``tau_abm_did`` (the same group-time estimator run on the
  simulator's rollout output, sampled firms only) must lie within the DiD's 95% CI
  widened by the simulator's Monte-Carlo SE, after de-attenuating the reported-scale
  DiD by the calibrated misclassification rates;
- magnitude: ``tau_abm`` must agree with ``tau_true`` in sign and be within 3x.

If either fails, the run is FLAGGED and ``reports/simulator_discrepancy.md`` is
written: where the simulator disagrees with those estimates, the simulator is wrong
first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from regworld.abm.model import load_observed_world
from regworld.abm.policies import STATIC_POLICIES
from regworld.abm.tensorized import rollout_tensorized
from regworld.causal.did import DidResult, estimate_did, group_time_att
from regworld.causal.estimate import PointEstimate, dml_onset
from regworld.causal.ground_truth import GroundTruthEffects, load_ground_truth
from regworld.data.ingest import read_panel_analysis
from regworld.data.store import read_observed
from regworld.rules import Theta
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)

# The historical regime's published lever settings (public regulatory facts, also
# registered as the `phased_targeted` static policy).
_HISTORICAL_POLICY = "phased_targeted"
_NEVER = 10_000  # onset beyond any horizon: the never-treated schedule


@dataclass(frozen=True)
class FourNumbers:
    """The gate's verdict plus every number a reviewer needs to check it."""

    tau_true: float
    tau_did_truth: float
    tau_abm: float
    tau_abm_mc_se: float
    tau_abm_did: float
    tau_abm_did_mc_se: float
    tau_qe: float
    tau_qe_ci: tuple[float, float]
    tau_qe_deattenuated: float
    tau_obs: float
    tau_obs_ci: tuple[float, float]
    report_scale: float
    sign_ok: bool
    magnitude_ok: bool
    did_agreement_ok: bool
    flagged: bool
    n_sim_seeds: int


def load_calibrated_theta(cfg: RegWorldConfig) -> tuple[Theta, float, float]:
    """Posterior-mean Theta from Stage 4's combined posterior, plus (q0, q1)."""
    import arviz as az

    path = Path(cfg.paths.root) / "calibration" / "posterior.nc"
    if not path.is_file():
        raise FileNotFoundError(f"calibrated posterior not found: {path}; run `make calibrate`")
    idata = az.from_netcdf(path)
    means: dict[str, float] = {}
    for name in idata.posterior.data_vars:
        if name in Theta.__dataclass_fields__:
            means[name] = float(np.asarray(idata.posterior[name]).mean())
    theta = Theta(**means)  # beta_capacity stays 0.0: latent z is never fitted
    return theta, theta.q0, theta.q1


def _treatment_schedule(
    cfg: RegWorldConfig, n_firms: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-firm 0-based onset quarters from the observed panel.

    Sampled firms carry their true cohort. Unsampled firms' regions are unknown
    (the registry has no region column), so their onsets are imputed by drawing
    from the observed cohort distribution — they only shape background spillovers;
    the simulated DiD is estimated on the sampled firms alone.
    """
    panel = read_observed(cfg, "firm_panel")
    firm_cohort = panel.group_by("firm_id").agg(pl.col("treatment_quarter").first())
    known = dict(
        zip(
            firm_cohort["firm_id"].to_list(),
            firm_cohort["treatment_quarter"].to_list(),
            strict=True,
        )
    )
    cohorts = np.asarray(list(known.values()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    schedule = np.empty(n_firms, dtype=np.int64)
    sampled_mask = np.zeros(n_firms, dtype=bool)
    for firm_id in range(n_firms):
        if firm_id in known:
            tq = int(known[firm_id])
            sampled_mask[firm_id] = True
        else:
            tq = int(rng.choice(cohorts))
        schedule[firm_id] = _NEVER if tq <= 0 else tq - 1  # 0-based onset
    return schedule, sampled_mask


def simulate_interventions(
    cfg: RegWorldConfig, theta: Theta, *, n_seeds: int = 8
) -> tuple[float, float, float, float]:
    """Run the calibrated simulator under do(rollout) and do(never).

    Returns ``(tau_abm, mc_se, tau_abm_did, did_mc_se)`` where ``tau_abm`` is the
    cell-averaged onset ATT against the never-treated counterfactual (the
    ``tau_true`` estimand) and ``tau_abm_did`` applies the panel's group-time
    estimator to the simulated rollout run (the ``tau_qe`` estimand).
    """
    world = load_observed_world(cfg)
    n_firms = world.firms.n
    quarters = cfg.observed_quarters
    schedule, sampled_mask = _treatment_schedule(cfg, n_firms, seed=cfg.seed + 71_000)
    never = np.full(n_firms, _NEVER, dtype=np.int64)
    levers = STATIC_POLICIES[_HISTORICAL_POLICY]
    sampled_ids = np.flatnonzero(sampled_mask)
    att_runs = np.empty(n_seeds)
    did_runs = np.empty(n_seeds)
    for k in range(n_seeds):
        seed = cfg.seed + 72_000 + k
        run_t = rollout_tensorized(
            cfg, world, theta, levers, seed=seed, quarters=quarters, treatment_start=schedule
        )
        run_c = rollout_tensorized(
            cfg, world, theta, levers, seed=seed, quarters=quarters, treatment_start=never
        )
        p_t = run_t.compliance_probabilities.detach().cpu().numpy()
        p_c = run_c.compliance_probabilities.detach().cpu().numpy()
        diff = p_t - p_c
        cells = []
        for t in range(quarters):
            post = schedule <= t
            if post.any():
                cells.append(float(diff[t, post].mean()))
        att_runs[k] = float(np.mean(cells)) if cells else 0.0
        rows_y, rows_q, rows_g = [], [], []
        for t in range(quarters - 1):
            rows_y.append(p_t[t, sampled_ids])
            rows_q.append(np.full(sampled_ids.size, t + 1, dtype=np.int64))
            cohort = np.where(schedule[sampled_ids] >= _NEVER, -1, schedule[sampled_ids] + 1)
            rows_g.append(cohort)
        did_runs[k], _, _ = group_time_att(
            np.concatenate(rows_y), np.concatenate(rows_q), np.concatenate(rows_g)
        )
    return (
        float(att_runs.mean()),
        float(att_runs.std(ddof=1) / np.sqrt(n_seeds)),
        float(did_runs.mean()),
        float(did_runs.std(ddof=1) / np.sqrt(n_seeds)),
    )


def run_gate(cfg: RegWorldConfig, *, n_sim_seeds: int = 8) -> FourNumbers:
    """Compute the four numbers and the verdict."""
    truth: GroundTruthEffects = load_ground_truth(cfg)
    panel = read_panel_analysis(cfg)
    did: DidResult = estimate_did(panel, seed=cfg.seed)
    dml: PointEstimate = dml_onset(panel, seed=cfg.seed)
    theta, q0, q1 = load_calibrated_theta(cfg)
    tau_abm, abm_se, tau_abm_did, abm_did_se = simulate_interventions(
        cfg, theta, n_seeds=n_sim_seeds
    )
    # The panel outcome is the *report*: misclassification shrinks effects by
    # (1 - q0 - q1). De-attenuate with the calibrated rates before comparing.
    scale = max(1.0 - q0 - q1, 0.5)
    tau_qe_deatt = did.att / scale
    half_width = (did.ci_high - did.ci_low) / 2.0 / scale
    did_agreement_ok = abs(tau_abm_did - tau_qe_deatt) <= half_width + 2.0 * abm_did_se
    sign_ok = np.sign(tau_abm) == np.sign(truth.onset_att) and tau_abm != 0.0
    ratio = abs(tau_abm) / max(abs(truth.onset_att), 1e-9)
    magnitude_ok = bool(sign_ok and (1.0 / 3.0) <= ratio <= 3.0)
    flagged = not (did_agreement_ok and magnitude_ok)
    return FourNumbers(
        tau_true=truth.onset_att,
        tau_did_truth=truth.did_truth,
        tau_abm=tau_abm,
        tau_abm_mc_se=abm_se,
        tau_abm_did=tau_abm_did,
        tau_abm_did_mc_se=abm_did_se,
        tau_qe=did.att,
        tau_qe_ci=(did.ci_low, did.ci_high),
        tau_qe_deattenuated=tau_qe_deatt,
        tau_obs=dml.estimate,
        tau_obs_ci=(dml.ci_low, dml.ci_high),
        report_scale=scale,
        sign_ok=bool(sign_ok),
        magnitude_ok=magnitude_ok,
        did_agreement_ok=bool(did_agreement_ok),
        flagged=flagged,
        n_sim_seeds=n_sim_seeds,
    )


def write_gate_outputs(cfg: RegWorldConfig, result: FourNumbers) -> list[Path]:
    """Persist four_numbers.json, plus the discrepancy report when FLAGGED."""
    out_dir = Path(cfg.paths.root) / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers_path = out_dir / "four_numbers.json"
    numbers_path.write_text(json.dumps(asdict(result), indent=2))
    paths = [numbers_path]
    if result.flagged:
        report = Path(cfg.paths.reports) / "simulator_discrepancy.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Simulator discrepancy (Stage 5f gate FLAGGED)\n\n"
            f"| quantity | value |\n|---|---|\n"
            f"| tau_true (sealed) | {result.tau_true:.4f} |\n"
            f"| tau_did_truth (sealed) | {result.tau_did_truth:.4f} |\n"
            f"| tau_abm | {result.tau_abm:.4f} (MC SE {result.tau_abm_mc_se:.4f}) |\n"
            f"| tau_abm_did | {result.tau_abm_did:.4f} (MC SE {result.tau_abm_did_mc_se:.4f}) |\n"
            f"| tau_qe (DiD) | {result.tau_qe:.4f} CI {result.tau_qe_ci} |\n"
            f"| tau_obs (DML) | {result.tau_obs:.4f} CI {result.tau_obs_ci} |\n\n"
            f"sign_ok={result.sign_ok} magnitude_ok={result.magnitude_ok} "
            f"did_agreement_ok={result.did_agreement_ok}\n\n"
            "Where the simulator disagrees with a credibly identified estimate, the "
            "simulator is wrong first. Per `causal.on_disagreement`, recalibrate with "
            "the DiD as a moment-matching penalty, or continue with warning banners.\n"
        )
        paths.append(report)
        log.warning("simulator gate FLAGGED; wrote %s", report)
    else:
        log.info("simulator gate PASSED")
    return paths
