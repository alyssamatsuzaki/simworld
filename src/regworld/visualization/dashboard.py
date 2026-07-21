"""Stage 15 (§10 Phase 7): the Streamlit operator dashboard.

The thing a policy team can actually operate: four lever sliders, a live
trajectory fan from the trained emulator (instant lookup for the static
policy grid, live inference off-grid), the Pareto frontier with the current
slider position marked, a backfire indicator, the non-compliance network
map, the sensitivity tornado, and — the one check the client actually cares
about — an out-of-distribution warning banner.

``ood_mahalanobis`` is a free function with no Streamlit dependency so it can
be unit-tested directly (``tests/test_visualization_contract.py``).

Streamlit execs this file as ``__main__`` under ``streamlit run``; importing
it as ``regworld.visualization.dashboard`` (or ``scripts.dashboard``) for
testing must not build any UI, so all Streamlit calls live inside ``main()``
behind the ``if __name__ == "__main__"`` guard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from regworld.types import RegWorldConfig, validate_config
from regworld.visualization._io import action_bounds, load_json

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "configs")
OOD_THRESHOLD = 3.0  # Mahalanobis distance beyond which the banner fires


def load_default_config(profile: str = "smoke") -> RegWorldConfig:
    """Compose+validate the default Hydra config without the ``@hydra.main`` decorator."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"profile={profile}"])
    return validate_config(cfg)


def ood_mahalanobis(action: np.ndarray, train_actions: np.ndarray) -> float:
    """Mahalanobis distance of ``action`` from the training action distribution.

    ``train_actions`` is ``(n_samples, n_levers)``; an in-distribution action
    (near the training mean, along the training covariance) scores near 0,
    a far-outside one scores large. Reusable and unit-testable in isolation
    (§15 acceptance: the client-critical check).
    """
    train_actions = np.asarray(train_actions, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    mean = train_actions.mean(axis=0)
    cov = np.cov(train_actions.T) + 1e-6 * np.eye(train_actions.shape[1])
    cov_inv = np.linalg.inv(cov)
    d = action - mean
    return float(np.sqrt(d @ cov_inv @ d))


def _fallback_train_actions(cfg: RegWorldConfig, n_samples: int = 512) -> np.ndarray:
    """Uniform draws over the action box — used only when no training corpus exists."""
    low, high = action_bounds()
    rng = np.random.default_rng(cfg.seed)
    return rng.uniform(low, high, size=(n_samples, low.shape[0]))


def _train_action_distribution(cfg: RegWorldConfig) -> np.ndarray:
    """Training action distribution behind the OOD banner.

    Prefers the real corpus of emulator training episodes
    (``artifacts/emulator/dataset``); falls back to a uniform draw over the
    action box (still leakage-safe — no answer-key access) when that corpus
    is not present, so the dashboard still renders on a cold clone.
    """
    try:
        from regworld.training.datamodule import EmulatorSequences

        train = EmulatorSequences(cfg, "train")
        return np.stack(
            [train.episode_arrays(e)["action"][1:].mean(axis=0) for e in train.episodes]
        )
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("training action corpus unavailable (%s); falling back to a uniform draw", exc)
        return _fallback_train_actions(cfg)


def _match_grid_policy(action: np.ndarray, tol: float = 0.03) -> str | None:
    """The static policy name whose lever vector matches ``action`` within ``tol``, if any."""
    from regworld.abm.policies import STATIC_POLICIES

    for name, levers in STATIC_POLICIES.items():
        if np.allclose(action, levers.as_array(), atol=tol):
            return name
    return None


def _load_model(cfg: RegWorldConfig) -> tuple[Any, dict[str, Any]] | None:
    from regworld.training.checkpoint import checkpoint_path, load_checkpoint

    path = checkpoint_path(cfg.paths.root, cfg.emulator.arch)
    if not path.is_file():
        return None
    model, meta = load_checkpoint(path)
    if "extras" not in meta:
        meta["extras"] = {}
    if "n_firms" not in meta["extras"]:
        meta["extras"]["n_firms"] = cfg.population.n_firms
    return model, meta


def _policy_fan(
    cfg: RegWorldConfig,
    model: Any,
    meta: dict[str, Any],
    action: np.ndarray,
    n_seeds: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the emulator forward under a constant action for several seeds.

    Returns (compliance, hhi), each ``(n_seeds, horizon)`` in natural units.
    """
    from regworld.environments.emulator_env import EmulatorEnv

    env = EmulatorEnv(cfg, model=model, meta=meta)
    compliance = np.zeros((n_seeds, horizon), dtype=np.float64)
    hhi = np.zeros((n_seeds, horizon), dtype=np.float64)
    action32 = action.astype(np.float32)
    for s in range(n_seeds):
        env.reset(seed=cfg.seed + 90_000 + s)
        terminated = truncated = False
        for t in range(horizon):
            if terminated or truncated:
                compliance[s, t] = compliance[s, t - 1] if t > 0 else np.nan
                hhi[s, t] = hhi[s, t - 1] if t > 0 else np.nan
                continue
            _, _, terminated, truncated, _ = env.step(action32)
            aggregates = np.asarray(env._aggregates, dtype=np.float64)
            compliance[s, t] = aggregates[0]
            hhi[s, t] = aggregates[2]
    env.close()
    return compliance, hhi


def _pareto_points(cfg: RegWorldConfig) -> Any | None:
    from regworld.visualization._io import load_cube

    return load_cube(cfg)


def main() -> None:  # pragma: no cover - exercised via `streamlit run`, not pytest
    import matplotlib.pyplot as plt
    import streamlit as st

    from regworld.abm.policies import STATIC_POLICIES
    from regworld.rules import PolicyLevers, backfire
    from regworld.training.datamodule import aggregate_to_outcome

    st.set_page_config(page_title="RegWorld operator dashboard", layout="wide")
    st.title("RegWorld — policy operator dashboard")

    profile = st.sidebar.selectbox("profile", ["smoke", "dev", "full"], index=0)
    cfg = load_default_config(profile)

    low, high = action_bounds()
    st.sidebar.header("Policy levers")
    enforcement = st.sidebar.slider("enforcement", float(low[0]), float(high[0]), 0.5)
    targeting = st.sidebar.slider("targeting", float(low[1]), float(high[1]), 0.0)
    phase_speed = st.sidebar.slider("phase-in speed", float(low[2]), float(high[2]), 0.5)
    subsidy = st.sidebar.slider("subsidy", float(low[3]), float(high[3]), 0.0)
    st.sidebar.caption(
        "The trained emulator's action space has 4 levers (enforcement, targeting, "
        "phase-in speed, subsidy); a fine-scale lever is not part of this action space."
    )
    action = np.array([enforcement, targeting, phase_speed, subsidy], dtype=np.float64)

    # --- OOD banner --------------------------------------------------------
    train_actions = _train_action_distribution(cfg)
    distance = ood_mahalanobis(action, train_actions)
    if distance > OOD_THRESHOLD:
        st.error(
            f"OUT OF DISTRIBUTION: Mahalanobis distance {distance:.2f} exceeds the "
            f"{OOD_THRESHOLD:.1f} warning threshold. The emulator is extrapolating "
            "beyond its training action distribution — treat this prediction as "
            "unreliable."
        )
    else:
        st.success(
            f"In distribution: Mahalanobis distance {distance:.2f} (threshold {OOD_THRESHOLD:.1f})."
        )

    loaded = _load_model(cfg)
    if loaded is None:
        st.warning(
            f"No trained emulator checkpoint at artifacts/emulator/{cfg.emulator.arch}/model.pt "
            "— run `make emulator` first. Slider controls are shown but predictions "
            "are unavailable."
        )
        return
    model, meta = loaded

    grid_name = _match_grid_policy(action)
    n_seeds = 20 if grid_name is not None else 6
    lookup_kind = (
        f"grid lookup ({grid_name})" if grid_name is not None else "live off-grid inference"
    )
    st.caption(f"Prediction source: {lookup_kind}")

    horizon = cfg.horizon_quarters
    compliance, hhi = _policy_fan(cfg, model, meta, action, n_seeds=n_seeds, horizon=horizon)
    quarters = np.arange(1, horizon + 1)

    col_fan, col_side = st.columns([2, 1])
    with col_fan:
        st.subheader("Trajectory fan (50/80/95% credible bands)")
        fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(10, 4))
        for lo_q, hi_q, alpha in ((0.025, 0.975, 0.15), (0.10, 0.90, 0.25), (0.25, 0.75, 0.35)):
            ax_c.fill_between(
                quarters,
                np.nanquantile(compliance, lo_q, axis=0),
                np.nanquantile(compliance, hi_q, axis=0),
                color="tab:blue",
                alpha=alpha,
            )
            ax_h.fill_between(
                quarters,
                np.nanquantile(hhi, lo_q, axis=0),
                np.nanquantile(hhi, hi_q, axis=0),
                color="tab:orange",
                alpha=alpha,
            )
        ax_c.plot(quarters, np.nanmedian(compliance, axis=0), color="tab:blue")
        ax_c.set_xlabel("quarter")
        ax_c.set_ylabel("compliance rate")
        ax_h.plot(quarters, np.nanmedian(hhi, axis=0), color="tab:orange")
        ax_h.set_xlabel("quarter")
        ax_h.set_ylabel("HHI")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    terminal_compliance = float(np.nanmedian(compliance[:, -1]))
    terminal_hhi = float(np.nanmedian(hhi[:, -1]))
    n_firms = int(meta["extras"]["n_firms"])
    baseline_agg = np.asarray(meta["initial"]["aggregate"], dtype=np.float64)
    terminal_agg = baseline_agg.copy()
    terminal_agg[0] = terminal_compliance
    terminal_agg[2] = terminal_hhi
    terminal_outcome = aggregate_to_outcome(terminal_agg, n_firms)
    baseline_outcome = aggregate_to_outcome(baseline_agg, n_firms)
    is_backfire = backfire(terminal_outcome, baseline_outcome)

    with col_side:
        st.subheader("Terminal outcome")
        st.metric("compliance rate", f"{terminal_compliance:.3f}")
        st.metric("HHI", f"{terminal_hhi:.1f}", delta=f"{terminal_hhi - baseline_agg[2]:+.1f}")
        st.metric("consumer surplus (baseline)", f"{baseline_agg[4]:.3f}")
        if is_backfire:
            st.error("BACKFIRE: compliance up, HHI up, consumer surplus down.")
        else:
            st.success("No backfire signature detected.")

    st.subheader("Pareto frontier — current slider position marked")
    cube = _pareto_points(cfg)
    if cube is not None and cube.height > 0:
        fig2, ax = plt.subplots(figsize=(7, 5))
        baseline_hhi = (
            float(cube.filter(cube["policy"] == "none")["hhi"].mean())
            if "none" in cube["policy"].unique().to_list()
            else float(cube["hhi"].mean())
        )
        for name in sorted(cube["policy"].unique().to_list()):
            rows = cube.filter(cube["policy"] == name)
            ax.scatter(
                rows["compliance_rate"].mean(),
                rows["hhi"].mean() - baseline_hhi,
                label=name,
                s=60,
            )
        ax.scatter(
            terminal_compliance,
            terminal_hhi - baseline_hhi,
            color="red",
            marker="*",
            s=250,
            label="current sliders",
            zorder=5,
        )
        ax.set_xlabel("terminal compliance rate")
        ax.set_ylabel("delta-HHI vs. no-intervention baseline")
        ax.legend(fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("No scenario cube found (`make ensemble`) — Pareto frontier unavailable.")

    st.subheader("Sensitivity tornado")
    indices = load_json(Path(cfg.paths.root) / "sensitivity" / "indices.json")
    if indices and "sobol" in indices and "ST" in indices["sobol"]:
        st_idx = indices["sobol"]["ST"]
        names = sorted(st_idx, key=lambda k: st_idx[k])
        fig3, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1))
        ax.barh(names, [st_idx[n] for n in names], color="tab:blue")
        ax.set_xlabel("Sobol total-effect index")
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("No sensitivity indices found (`make sensitivity`).")

    st.subheader("Non-compliance network map")
    from regworld.visualization.interactive import network_diffusion_figure

    net_fig = network_diffusion_figure(cfg)
    if net_fig is not None:
        st.plotly_chart(net_fig, use_container_width=True)
    else:
        st.info("No network/emulator artifacts found for the non-compliance map.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Static policy grid: " + ", ".join(STATIC_POLICIES) + ".")
    st.sidebar.caption(f"Current levers as PolicyLevers: {PolicyLevers(*action.tolist())}")


if __name__ == "__main__":
    main()
