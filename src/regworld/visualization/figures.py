"""Stage 15 (§10 Phase 7): the thirteen Matplotlib paper figures.

Publication defaults, no interactivity — mirrors ``scripts/eval_emulator.py``'s
``_figures`` helper (``matplotlib.use("Agg")``, ``plt.subplots``, ``dpi=150``,
explicit ``plt.close``). Every figure function reads its own artifact(s) and
degrades gracefully: a missing input logs a warning and the function returns
``None`` rather than raising, so one absent artifact never takes down the
other twelve. ``make_all_figures`` is the single entry point the driver calls.

Leakage firewall (PLAN.md §1): nothing here imports ``regworld.dgp`` or reads
the sealed answer-key tree. Where a figure needs a "true" value for
comparison (parameter recovery), it reads the comparison that
``regworld.evaluation.parameter_recovery`` already computed and wrote to
``reports/eval/metrics.json`` — never the answer key directly.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from regworld.types import RegWorldConfig
from regworld.visualization._io import eval_metrics, load_cube, load_json

log = logging.getLogger(__name__)


def _fig_dir(cfg: RegWorldConfig) -> Path:
    out = Path(cfg.paths.reports) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _rng(cfg: RegWorldConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng(cfg.seed + salt)


def _save(fig: Any, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _fmt_pm(value: float | None, se: float | None) -> str:
    if value is None or se is None:
        return "n/a"
    return f"{value:.4f} ± {1.96 * se:.4f}"


def _fmt_ci(ci: list[float] | None) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


# --------------------------------------------------------------------- fig 1
def fig_four_numbers(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 1 — the four-number causal table (tau_true, tau_abm, tau_qe, tau_obs)."""
    path = Path(cfg.paths.root) / "causal" / "four_numbers.json"
    data = load_json(path)
    if data is None:
        return None

    rows = [
        ("tau_true", data.get("tau_true"), "n/a (do() ground truth)"),
        ("tau_abm", data.get("tau_abm"), _fmt_pm(data.get("tau_abm"), data.get("tau_abm_mc_se"))),
        ("tau_qe", data.get("tau_qe"), _fmt_ci(data.get("tau_qe_ci"))),
        ("tau_obs", data.get("tau_obs"), _fmt_ci(data.get("tau_obs_ci"))),
    ]
    cell_text = [
        [name, f"{value:.4f}" if isinstance(value, int | float) else "n/a", note]
        for name, value, note in rows
    ]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["estimator", "value", "uncertainty"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    flagged = bool(data.get("flagged", False))
    status = "FLAGGED" if flagged else "gate OK"
    color = "tab:red" if flagged else "tab:green"
    ax.set_title(f"The four-number causal table — {status}", color=color, fontweight="bold")
    return _save(fig, fig_dir / "fig01_four_numbers.png")


# --------------------------------------------------------------------- fig 2
def fig_parameter_recovery(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 2 — posterior marginals with true values overlaid.

    The comparison rows (parameter, true value, posterior mean) come from
    ``reports/eval/metrics.json['parameter_recovery']``, already graded by
    the evaluation suite against the answer key; this module never touches
    that key directly. The marginal draws come from the calibration NetCDF.
    """
    metrics = eval_metrics(cfg)
    posterior_path = Path(cfg.paths.root) / "calibration" / "micro_posterior.nc"
    if metrics is None or not posterior_path.is_file():
        log.warning("fig2 skipped: metrics.json or micro_posterior.nc missing")
        return None
    recovery = metrics.get("parameter_recovery")
    if not isinstance(recovery, dict) or "per_parameter" not in recovery:
        log.warning("fig2 skipped: no parameter_recovery.per_parameter in metrics.json")
        return None

    import arviz as az

    idata = az.from_netcdf(posterior_path)
    rows = [
        row
        for row in recovery["per_parameter"]
        if isinstance(row, dict) and row.get("parameter") in idata.posterior.data_vars
    ]
    if not rows:
        log.warning("fig2 skipped: no graded parameters present in the posterior")
        return None

    n = len(rows)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, row in zip(flat_axes, rows, strict=False):
        name = row["parameter"]
        draws = np.asarray(idata.posterior[name]).reshape(-1)
        ax.hist(draws, bins=30, density=True, color="tab:blue", alpha=0.6)
        true_value = row.get("true")
        if isinstance(true_value, int | float):
            ax.axvline(float(true_value), color="tab:red", ls="--", lw=1.5, label="true")
        mean_value = row.get("posterior_mean")
        if isinstance(mean_value, int | float):
            ax.axvline(float(mean_value), color="black", ls=":", lw=1.2, label="posterior mean")
        covers = row.get("hdi_90_covers")
        ax.set_title(f"{name}\n(90% HDI covers: {covers})", fontsize=9)
        ax.legend(fontsize=6)
    for ax in flat_axes[n:]:
        ax.axis("off")
    variant = recovery.get("dgp_variant", "?")
    fig.suptitle(f"Parameter recovery ({variant} world)", fontweight="bold")
    return _save(fig, fig_dir / "fig02_parameter_recovery.png")


# --------------------------------------------------------------------- fig 3
def fig_arviz_diagnostics(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 3 — ArviZ diagnostics panel: trace, energy, pair on beta_peer/beta_assoc."""
    posterior_path = Path(cfg.paths.root) / "calibration" / "micro_posterior.nc"
    if not posterior_path.is_file():
        log.warning("fig3 skipped: %s missing", posterior_path)
        return None

    import arviz as az

    idata = az.from_netcdf(posterior_path)
    var_names = [v for v in ("beta_peer", "beta_assoc") if v in idata.posterior.data_vars]
    if len(var_names) < 2:
        log.warning("fig3 skipped: beta_peer/beta_assoc not both present in the posterior")
        return None

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2)
    trace_axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    az.plot_trace(idata, var_names=var_names, axes=trace_axes, compact=True)

    ax_energy = fig.add_subplot(gs[2, 0])
    has_energy = "sample_stats" in idata.groups() and "energy" in idata.sample_stats.data_vars
    if has_energy:
        az.plot_energy(idata, ax=ax_energy)
    else:
        ax_energy.axis("off")
        ax_energy.text(0.5, 0.5, "no energy stats in this posterior", ha="center", va="center")

    ax_pair = fig.add_subplot(gs[2, 1])
    az.plot_pair(idata, var_names=var_names, kind="scatter", ax=ax_pair)

    fig.suptitle(
        "ArviZ diagnostics: trace / energy / pair (beta_peer, beta_assoc)", fontweight="bold"
    )
    return _save(fig, fig_dir / "fig03_arviz_diagnostics.png")


# --------------------------------------------------------------------- fig 4
def fig_event_study(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 4 — event study from the staggered rollout, with pre-trends."""
    path = Path(cfg.paths.root) / "causal" / "causal_estimates.json"
    data = load_json(path)
    if data is None or "did" not in data:
        log.warning("fig4 skipped: no 'did' block in causal_estimates.json")
        return None
    did = data["did"]
    times = did.get("event_times")
    coefs = did.get("event_coefficients")
    if not times or not coefs or len(times) != len(coefs):
        log.warning("fig4 skipped: event_times/event_coefficients missing or mismatched")
        return None

    se = did.get("se")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = [se] * len(times) if isinstance(se, int | float) else None
    ax.errorbar(times, coefs, yerr=yerr, fmt="o-", capsize=3, color="tab:blue")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.axvline(-0.5, color="black", ls=":", lw=1, label="policy onset")
    ax.set_xlabel("event time (quarters relative to rollout onset)")
    ax.set_ylabel("DiD coefficient")
    pretrend = did.get("pretrend_max_abs")
    pretrend_str = f"{pretrend:.3f}" if isinstance(pretrend, int | float) else "n/a"
    ax.set_title(f"Event study (max |pre-trend coef| = {pretrend_str})")
    ax.legend()
    return _save(fig, fig_dir / "fig04_event_study.png")


# --------------------------------------------------------------------- fig 5
def fig_emulator_error_vs_horizon(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 5 — one-step/H-step error vs horizon, GNN vs flat vs GRU ablation."""
    metrics = eval_metrics(cfg)
    if metrics is None:
        return None
    predictive = metrics.get("predictive")
    ablations = metrics.get("ablations")
    if not isinstance(predictive, dict) or "compliance_drift" not in predictive:
        log.warning("fig5 skipped: metrics.json has no predictive.compliance_drift")
        return None

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
    drift = predictive["compliance_drift"]
    persistence = predictive["persistence_drift"]
    ks = sorted(int(k[1:]) for k in drift)
    ax_left.plot(ks, [drift[f"k{k}"] for k in ks], "o-", label=f"{cfg.emulator.arch}")
    ax_left.plot(ks, [persistence[f"k{k}"] for k in ks], "s--", color="grey", label="persistence")
    ax_left.axhline(0.10, color="grey", lw=0.8, ls=":")
    ax_left.set_xlabel("imagination horizon k (quarters)")
    ax_left.set_ylabel("compliance MAE")
    ax_left.set_title("k-step open-loop drift")
    ax_left.legend()

    ablation_table = ablations.get("table") if isinstance(ablations, dict) else None
    if isinstance(ablation_table, list) and ablation_table:
        table = ablation_table
        archs = [str(row["arch"]) for row in table]
        one_step = [float(row["one_step_compliance_mae"]) for row in table]
        open_loop = [float(row["open_loop_compliance_mae"]) for row in table]
        x = np.arange(len(archs))
        width = 0.35
        ax_right.bar(x - width / 2, one_step, width, label="one-step MAE")
        ax_right.bar(x + width / 2, open_loop, width, label="open-loop MAE")
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(archs)
        ax_right.set_ylabel("compliance MAE")
        ax_right.set_title("architecture ablation")
        ax_right.legend()
    else:
        ax_right.axis("off")
        ax_right.text(0.5, 0.5, "no ablation table available", ha="center", va="center")

    fig.suptitle("Emulator error vs horizon and architecture", fontweight="bold")
    return _save(fig, fig_dir / "fig05_emulator_error_vs_horizon.png")


# --------------------------------------------------------------------- fig 6
def fig_imagined_vs_real(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 6 — imagined-vs-real rollouts, 6 sampled trajectories side by side."""
    checkpoint = Path(cfg.paths.root) / "emulator" / cfg.emulator.arch / "model.pt"
    if not checkpoint.is_file():
        log.warning("fig6 skipped: %s missing", checkpoint)
        return None

    import torch

    from regworld.evaluation.harness import load_context, open_loop_natural

    try:
        ctx = load_context(cfg)
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("fig6 skipped: failed to load evaluation context: %s", exc)
        return None

    batch = ctx.batch
    steps = batch["firm"].shape[1]
    horizon = steps - 1
    agg, _, start = open_loop_natural(
        ctx.model, batch, burn_in=1, horizon=horizon, generator=torch.Generator().manual_seed(6)
    )
    real = batch["aggregate"][:, start + 1 : start + 1 + horizon, 0].numpy()
    imagined = np.clip(agg[..., 0], 0.0, 1.0)

    n_episodes = real.shape[0]
    n_show = min(6, n_episodes)
    rng = _rng(cfg, salt=6_000)
    chosen = rng.choice(n_episodes, size=n_show, replace=False)
    quarters = np.arange(1, horizon + 1)

    ncols = min(3, n_show)
    nrows = math.ceil(n_show / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, ep in zip(flat_axes, chosen, strict=False):
        ax.plot(quarters, real[ep], "o-", label="real (held-out ABM)", color="tab:blue")
        ax.plot(quarters, imagined[ep], "s--", label="imagined (emulator)", color="tab:orange")
        ax.set_title(f"episode {ctx.episodes[ep]}", fontsize=9)
        ax.set_xlabel("quarter")
        ax.set_ylabel("compliance rate")
    flat_axes[0].legend(fontsize=8)
    for ax in flat_axes[n_show:]:
        ax.axis("off")
    fig.suptitle("Imagined vs. real rollouts (6 held-out episodes)", fontweight="bold")
    return _save(fig, fig_dir / "fig06_imagined_vs_real.png")


# --------------------------------------------------------------------- fig 7
def fig_calibration_curve(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 7 — calibration curve and 90% interval coverage."""
    metrics = eval_metrics(cfg)
    if metrics is None:
        return None
    calibration = metrics.get("calibration")
    if not isinstance(calibration, dict) or "reliability_diagram" not in calibration:
        log.warning("fig7 skipped: metrics.json has no calibration.reliability_diagram")
        return None

    bins = [b for b in calibration["reliability_diagram"] if b.get("confidence") is not None]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    if bins:
        ax_left.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax_left.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            label=f"ECE {calibration.get('ece', float('nan')):.3f}",
        )
        ax_left.set_xlabel("predicted compliance probability")
        ax_left.set_ylabel("empirical frequency")
        ax_left.set_title("one-step node reliability")
        ax_left.legend()
    else:
        ax_left.axis("off")
        ax_left.text(0.5, 0.5, "no reliability bins available", ha="center", va="center")

    levels = [50, 80, 90, 95]
    empirical = [calibration.get(f"coverage_{lv}") for lv in levels]
    have = [(lv, e) for lv, e in zip(levels, empirical, strict=True) if isinstance(e, int | float)]
    if have:
        lv_vals, emp_vals = zip(*have, strict=True)
        nominal = [lv / 100 for lv in lv_vals]
        x = np.arange(len(lv_vals))
        width = 0.35
        ax_right.bar(x - width / 2, nominal, width, label="nominal", color="grey")
        ax_right.bar(x + width / 2, emp_vals, width, label="empirical", color="tab:blue")
        ax_right.set_xticks(x)
        ax_right.set_xticklabels([f"{lv}%" for lv in lv_vals])
        ax_right.set_ylabel("coverage")
        ax_right.set_title("predictive interval coverage")
        ax_right.legend()
    else:
        ax_right.axis("off")
        ax_right.text(0.5, 0.5, "no coverage levels available", ha="center", va="center")

    fig.suptitle("Calibration", fontweight="bold")
    return _save(fig, fig_dir / "fig07_calibration_coverage.png")


# --------------------------------------------------------------------- fig 8
def _rollout_trajectory(
    env: Any, action: np.ndarray, horizon: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One full-horizon rollout; returns (compliance, hhi) per quarter, quarter 0 excluded."""
    env.reset(seed=seed)
    compliance = np.zeros(horizon, dtype=np.float64)
    hhi = np.zeros(horizon, dtype=np.float64)
    terminated = truncated = False
    for t in range(horizon):
        if terminated or truncated:
            compliance[t] = compliance[t - 1] if t > 0 else np.nan
            hhi[t] = hhi[t - 1] if t > 0 else np.nan
            continue
        _, _, terminated, truncated, _ = env.step(action)
        aggregates = np.asarray(env._aggregates, dtype=np.float64)
        compliance[t] = aggregates[0]
        hhi[t] = aggregates[2]
    return compliance, hhi


def fig_trajectory_fans(cfg: RegWorldConfig, fig_dir: Path, n_seeds: int = 12) -> Path | None:
    """Fig 8 — compliance and HHI trajectory fans over the horizon, per policy.

    The scenario cube (Stage 11) stores terminal outcomes only, so the
    per-quarter fan is built by re-running the trained emulator forward under
    each static policy for a handful of seeds — the same machinery
    ``regworld.ensemble.cube`` uses per cell, just recording every quarter
    instead of only the last one.
    """
    checkpoint = Path(cfg.paths.root) / "emulator" / cfg.emulator.arch / "model.pt"
    if not checkpoint.is_file():
        log.warning("fig8 skipped: %s missing", checkpoint)
        return None

    from regworld.abm.policies import STATIC_POLICIES
    from regworld.environments.emulator_env import EmulatorEnv
    from regworld.training.checkpoint import checkpoint_path, load_checkpoint

    try:
        model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("fig8 skipped: checkpoint failed to load: %s", exc)
        return None
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms

    horizon = cfg.horizon_quarters
    policies = list(STATIC_POLICIES.items())
    fig, axes = plt.subplots(2, len(policies), figsize=(3.2 * len(policies), 6), squeeze=False)
    quarters = np.arange(1, horizon + 1)
    bands = ((0.05, 0.95, 0.15), (0.10, 0.90, 0.25), (0.25, 0.75, 0.35))

    for col, (name, levers) in enumerate(policies):
        env = EmulatorEnv(cfg, model=model, meta=meta)
        action = levers.as_array().astype(np.float32)
        compliance_runs = np.zeros((n_seeds, horizon), dtype=np.float64)
        hhi_runs = np.zeros((n_seeds, horizon), dtype=np.float64)
        for s in range(n_seeds):
            seed = cfg.seed + 8_000 + s
            compliance_runs[s], hhi_runs[s] = _rollout_trajectory(env, action, horizon, seed)
        env.close()

        for lo_q, hi_q, alpha in bands:
            lo = np.nanquantile(compliance_runs, lo_q, axis=0)
            hi = np.nanquantile(compliance_runs, hi_q, axis=0)
            axes[0, col].fill_between(quarters, lo, hi, color="tab:blue", alpha=alpha)
        axes[0, col].plot(quarters, np.nanmedian(compliance_runs, axis=0), color="tab:blue")
        axes[0, col].set_title(name, fontsize=9)
        axes[0, col].set_ylim(0, 1)
        if col == 0:
            axes[0, col].set_ylabel("compliance rate")

        for lo_q, hi_q, alpha in bands:
            lo = np.nanquantile(hhi_runs, lo_q, axis=0)
            hi = np.nanquantile(hhi_runs, hi_q, axis=0)
            axes[1, col].fill_between(quarters, lo, hi, color="tab:orange", alpha=alpha)
        axes[1, col].plot(quarters, np.nanmedian(hhi_runs, axis=0), color="tab:orange")
        axes[1, col].set_xlabel("quarter")
        if col == 0:
            axes[1, col].set_ylabel("HHI")

    fig.suptitle(
        f"Trajectory fans across {n_seeds} seeds (50/80/90% credible bands)", fontweight="bold"
    )
    return _save(fig, fig_dir / "fig08_trajectory_fans.png")


# --------------------------------------------------------------------- fig 9
def fig_pareto_frontier(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 9 — the Pareto frontier: terminal compliance vs delta-HHI per policy."""
    cube = load_cube(cfg)
    if cube is None or cube.height == 0:
        return None

    baseline_hhi = None
    if "none" in cube["policy"].unique().to_list():
        baseline_hhi = float(cube.filter(cube["policy"] == "none")["hhi"].mean())

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("RdYlGn_r")
    policies = cube["policy"].unique().sort().to_list()
    for name in policies:
        rows = cube.filter(cube["policy"] == name)
        compliance = rows["compliance_rate"].to_numpy()
        hhi = rows["hhi"].to_numpy()
        delta_hhi = hhi - baseline_hhi if baseline_hhi is not None else hhi - hhi.mean()
        backfire_rate = float(rows["backfire"].cast(bool).to_numpy().mean())
        mean_point = (float(compliance.mean()), float(delta_hhi.mean()))
        color = cmap(min(max(backfire_rate, 0.0), 1.0))

        if len(compliance) >= 2 and np.std(compliance) > 0 and np.std(delta_hhi) > 0:
            cov = np.cov(np.stack([compliance, delta_hhi]))
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            width, height = 2 * np.sqrt(np.maximum(eigvals, 0.0))
            ellipse = Ellipse(
                mean_point,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                alpha=0.25,
                edgecolor=color,
            )
            ax.add_patch(ellipse)
        ax.scatter(*mean_point, color=color, s=80, edgecolor="black", zorder=3)
        ax.annotate(name, mean_point, textcoords="offset points", xytext=(6, 6), fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="backfire probability")
    ax.set_xlabel("terminal compliance rate")
    ax.set_ylabel("delta-HHI vs. no-intervention baseline")
    ax.set_title("Pareto frontier: compliance vs. market concentration")
    return _save(fig, fig_dir / "fig09_pareto_frontier.png")


# -------------------------------------------------------------------- fig 10
def fig_sensitivity_tornado(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 10 — Sobol total-effect (ST) sensitivity tornado."""
    path = Path(cfg.paths.root) / "sensitivity" / "indices.json"
    data = load_json(path)
    if data is None or "sobol" not in data:
        log.warning("fig10 skipped: no 'sobol' block in indices.json")
        return None
    sobol = data["sobol"]
    st = sobol.get("ST")
    s1 = sobol.get("S1")
    if not isinstance(st, dict):
        log.warning("fig10 skipped: sobol.ST missing")
        return None

    names = sorted(st, key=lambda k: st[k])
    st_vals = [float(st[n]) for n in names]
    s1_vals = [float(s1[n]) for n in names] if isinstance(s1, dict) else None

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, st_vals, color="tab:blue", label="ST (total effect)")
    if s1_vals is not None:
        ax.scatter(s1_vals, y, color="black", zorder=3, label="S1 (first order)")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sobol index")
    ax.set_title("Sensitivity tornado (emulator Sobol indices)")
    ax.legend()
    return _save(fig, fig_dir / "fig10_sensitivity_tornado.png")


# -------------------------------------------------------------------- fig 11
def fig_noncompliance_network(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 11 — non-compliance concentration over the supply network."""
    checkpoint = Path(cfg.paths.root) / "emulator" / cfg.emulator.arch / "model.pt"
    edges_path = Path(cfg.paths.data) / "observed" / "graphs" / "supply_edges.parquet"
    registry_path = Path(cfg.paths.data) / "observed" / "firm_registry.parquet"
    if not checkpoint.is_file() or not edges_path.is_file() or not registry_path.is_file():
        log.warning("fig11 skipped: checkpoint, supply edges, or firm registry missing")
        return None

    import networkx as nx
    import polars as pl
    import torch

    from regworld.evaluation.harness import load_context, open_loop_natural

    try:
        ctx = load_context(cfg)
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("fig11 skipped: failed to load evaluation context: %s", exc)
        return None

    batch = ctx.batch
    mid = max(1, batch["firm"].shape[1] // 2)
    _, node_probs, _ = open_loop_natural(
        ctx.model, batch, burn_in=mid, horizon=1, generator=torch.Generator().manual_seed(11)
    )
    alive = batch["firm"][:, mid, :, 1].numpy() > 0.5
    prob = node_probs[:, 0]  # (B, N)
    weighted_prob = np.where(alive, prob, np.nan)
    with warnings.catch_warnings():
        # A firm dead in every held-out episode at this quarter has no alive
        # observation to average; nanmean warns on the all-NaN column, and
        # the resulting NaN is handled explicitly below (falls back to 0.0).
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_compliance_prob = np.nanmean(weighted_prob, axis=0)  # (N,) node index order
    risk = 1.0 - mean_compliance_prob

    edges = pl.read_parquet(edges_path)
    registry = pl.read_parquet(registry_path)
    graph = nx.DiGraph()
    for firm_id, size_decile in zip(
        registry["firm_id"].to_list(), registry["size_decile"].to_list(), strict=True
    ):
        graph.add_node(int(firm_id), size_decile=int(size_decile))
    for src, dst in zip(edges["src"].to_list(), edges["dst"].to_list(), strict=True):
        graph.add_edge(int(src), int(dst))

    n_nodes = len(risk)
    node_colors = [
        float(risk[n]) if 0 <= n < n_nodes and np.isfinite(risk[n]) else 0.0 for n in graph.nodes
    ]
    node_sizes = [30.0 + 12.0 * float(graph.nodes[n].get("size_decile", 1)) for n in graph.nodes]

    fig, ax = plt.subplots(figsize=(8, 7))
    layout = nx.spring_layout(graph, seed=cfg.seed)
    nx.draw_networkx_edges(graph, layout, ax=ax, alpha=0.15, arrows=False)
    nodes = nx.draw_networkx_nodes(
        graph,
        layout,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(nodes, ax=ax, label="P(non-compliant)")
    ax.set_title(
        "Non-compliance concentration over the supply network\n(node size ~ firm size decile)"
    )
    ax.axis("off")
    return _save(fig, fig_dir / "fig11_noncompliance_network.png")


# -------------------------------------------------------------------- fig 12
def fig_policy_comparison_j(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 12 — J_emulator vs J_ABM per policy: the exploitation gap."""
    metrics = eval_metrics(cfg)
    cube = load_cube(cfg)
    if metrics is None or cube is None:
        return None
    planning = metrics.get("planning_utility")
    if not isinstance(planning, dict) or not isinstance(planning.get("policies"), dict):
        log.warning("fig12 skipped: metrics.json has no planning_utility.policies")
        return None

    j_abm = {name: float(row["mean_return"]) for name, row in planning["policies"].items()}
    j_emulator = {
        name: float(cube.filter(cube["policy"] == name)["reward"].mean())
        for name in cube["policy"].unique().to_list()
    }
    common = sorted(set(j_abm) & set(j_emulator))
    if not common:
        log.warning("fig12 skipped: no policy names shared between the cube and planning_utility")
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(common))
    width = 0.35
    ax.bar(
        x - width / 2,
        [j_emulator[n] for n in common],
        width,
        label="J_emulator",
        color="tab:orange",
    )
    ax.bar(x + width / 2, [j_abm[n] for n in common], width, label="J_ABM", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=20, ha="right")
    ax.set_ylabel("regulator objective J")
    ax.set_title("Policy comparison: emulator vs. ABM (exploitation gap)")
    ax.legend()
    return _save(fig, fig_dir / "fig12_policy_comparison_j.png")


# -------------------------------------------------------------------- fig 13
def fig_ood_degradation(cfg: RegWorldConfig, fig_dir: Path) -> Path | None:
    """Fig 13 — emulator error vs. distance from the training action distribution."""
    metrics = eval_metrics(cfg)
    if metrics is None:
        return None
    ood = metrics.get("ood")
    if not isinstance(ood, dict) or "heldout_mean_error" not in ood:
        log.warning("fig13 skipped: metrics.json has no ood block")
        return None

    distances = ood.get("heldout_distances") or []
    in_range_distance = float(np.mean(distances)) if distances else 0.0
    in_range_error = float(ood["heldout_mean_error"])
    extreme_distance = ood.get("enforcement_1p5_mahalanobis")
    extreme_error = ood.get("enforcement_1p5_error")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        [in_range_distance],
        [in_range_error],
        color="tab:blue",
        s=90,
        label="held-out (in-range)",
    )
    if isinstance(extreme_distance, int | float) and isinstance(extreme_error, int | float):
        ax.scatter(
            [extreme_distance],
            [extreme_error],
            color="tab:red",
            s=90,
            label="enforcement 1.5x",
        )
        ax.plot(
            [in_range_distance, extreme_distance],
            [in_range_error, extreme_error],
            "k--",
            lw=1,
        )
    corr = ood.get("heldout_error_vs_mahalanobis_spearman")
    corr_str = f"{corr:.2f}" if isinstance(corr, int | float) else "n/a"
    ax.set_xlabel("Mahalanobis distance from training action distribution")
    ax.set_ylabel("compliance MAE")
    ax.set_title(f"Fig 13: OOD degradation (Spearman rho = {corr_str})")
    ax.legend()
    return _save(fig, fig_dir / "fig13_ood_degradation.png")


FIGURE_FUNCS: tuple[Any, ...] = (
    fig_four_numbers,
    fig_parameter_recovery,
    fig_arviz_diagnostics,
    fig_event_study,
    fig_emulator_error_vs_horizon,
    fig_imagined_vs_real,
    fig_calibration_curve,
    fig_trajectory_fans,
    fig_pareto_frontier,
    fig_sensitivity_tornado,
    fig_noncompliance_network,
    fig_policy_comparison_j,
    fig_ood_degradation,
)


def make_all_figures(cfg: RegWorldConfig) -> list[Path]:
    """Build every available figure into ``reports/figures/``; return written paths."""
    fig_dir = _fig_dir(cfg)
    written: list[Path] = []
    for func in FIGURE_FUNCS:
        try:
            out = func(cfg, fig_dir)
        except Exception:
            log.exception("%s raised; skipping this figure", func.__name__)
            continue
        if out is not None:
            written.append(out)
    log.info("wrote %d/%d figures to %s", len(written), len(FIGURE_FUNCS), fig_dir)
    return written
