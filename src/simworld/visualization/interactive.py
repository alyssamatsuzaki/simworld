"""Stage 15 (§10 Phase 7): Plotly exploration widgets.

Trajectory fans, a latent-space PCA projection, and the network diffusion
map, as interactive HTML. Every function returns a Plotly ``Figure`` (or
``None`` if its input artifact is missing) so callers — tests, the
dashboard, ``write_all_interactive`` below — can compose them freely.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from simworld.types import SimWorldConfig
from simworld.visualization._io import load_cube

log = logging.getLogger(__name__)


def _rng(cfg: SimWorldConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng(cfg.seed + salt)


def trajectory_fans_figure(cfg: SimWorldConfig) -> Any | None:
    """Terminal compliance/HHI distribution per policy, from the scenario cube.

    The cube (``artifacts/ensemble/cube.parquet``) holds terminal outcomes
    per (policy, posterior-draw, seed) cell; this renders their spread as a
    box-per-policy fan, which is what is actually available interactively
    without re-running the emulator on every page load.
    """
    cube = load_cube(cfg)
    if cube is None or cube.height == 0:
        return None

    import plotly.graph_objects as go

    fig = go.Figure()
    for name in sorted(cube["policy"].unique().to_list()):
        rows = cube.filter(cube["policy"] == name)
        fig.add_trace(go.Box(y=rows["compliance_rate"].to_numpy(), name=name, boxmean=True))
    fig.update_layout(
        title="Terminal compliance rate by policy (posterior x seed spread)",
        yaxis_title="compliance rate",
        template="plotly_white",
    )
    return fig


def latent_pca_figure(cfg: SimWorldConfig) -> Any | None:
    """PCA of firm-node latents, coloured by compliance regime.

    Runs one imagination step from the trained emulator on the held-out
    initial frame to obtain the micro-recurrence's hidden state per firm
    node, then projects it to 2D. Degrades to ``None`` if no checkpoint is
    available.
    """
    checkpoint = Path(cfg.paths.root) / "emulator" / cfg.emulator.arch / "model.pt"
    if not checkpoint.is_file():
        log.warning("latent PCA skipped: %s missing", checkpoint)
        return None

    import torch
    from sklearn.decomposition import PCA

    from simworld.abm.policies import STATIC_POLICIES
    from simworld.training.checkpoint import checkpoint_path, load_checkpoint

    try:
        model, meta = load_checkpoint(checkpoint_path(cfg.paths.root, cfg.emulator.arch))
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("latent PCA skipped: checkpoint failed to load: %s", exc)
        return None

    initial = {k: v.float() for k, v in meta["initial"].items()}
    generator = torch.Generator().manual_seed(cfg.seed)
    state = model.initial_state(
        initial["firm"].unsqueeze(0),
        initial["segment"].unsqueeze(0),
        initial["aggregate"].unsqueeze(0),
        generator,
    )
    action = torch.as_tensor(
        next(iter(STATIC_POLICIES.values())).as_array(), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        new_state, decoded = model.imagine_step(state, action, generator)

    node_hidden = new_state.node_hidden
    if node_hidden is None:
        log.warning("latent PCA skipped: this architecture has no node-level hidden state")
        return None
    latents = node_hidden.numpy()  # (N_firm, H)
    if latents.shape[0] < 2:
        return None
    compliance_prob = decoded.node_probs[0].numpy()
    regime = np.where(compliance_prob >= 0.5, "compliant", "non-compliant")

    n_components = min(2, latents.shape[1])
    coords = PCA(n_components=n_components, random_state=cfg.seed).fit_transform(latents)
    if n_components == 1:
        coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)

    import plotly.express as px

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=regime,
        labels={"x": "PC1", "y": "PC2", "color": "compliance regime"},
        title="Firm-node latent space (PCA), coloured by compliance regime",
    )
    fig.update_layout(template="plotly_white")
    return fig


def network_diffusion_figure(cfg: SimWorldConfig) -> Any | None:
    """Interactive supply-network map, node colour = predicted non-compliance risk."""
    checkpoint = Path(cfg.paths.root) / "emulator" / cfg.emulator.arch / "model.pt"
    edges_path = Path(cfg.paths.data) / "observed" / "graphs" / "supply_edges.parquet"
    registry_path = Path(cfg.paths.data) / "observed" / "firm_registry.parquet"
    if not checkpoint.is_file() or not edges_path.is_file() or not registry_path.is_file():
        log.warning("network diffusion map skipped: required artifacts missing")
        return None

    import networkx as nx
    import polars as pl
    import torch

    from simworld.evaluation.harness import load_context, open_loop_natural

    try:
        ctx = load_context(cfg)
    except Exception as exc:  # pragma: no cover - artifact-dependent
        log.warning("network diffusion map skipped: failed to load evaluation context: %s", exc)
        return None

    batch = ctx.batch
    mid = max(1, batch["firm"].shape[1] // 2)
    _, node_probs, _ = open_loop_natural(
        ctx.model,
        batch,
        burn_in=mid,
        horizon=1,
        generator=torch.Generator().manual_seed(111),
    )
    alive = batch["firm"][:, mid, :, 1].numpy() > 0.5
    prob = np.where(alive, node_probs[:, 0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        risk = 1.0 - np.nanmean(prob, axis=0)

    edges = pl.read_parquet(edges_path)
    registry = pl.read_parquet(registry_path)
    graph = nx.DiGraph()
    for firm_id in registry["firm_id"].to_list():
        graph.add_node(int(firm_id))
    for src, dst in zip(edges["src"].to_list(), edges["dst"].to_list(), strict=True):
        graph.add_edge(int(src), int(dst))
    layout = nx.spring_layout(graph, seed=cfg.seed)

    import plotly.graph_objects as go

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in graph.edges():
        edge_x += [layout[u][0], layout[v][0], None]
        edge_y += [layout[u][1], layout[v][1], None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line={"width": 0.5, "color": "#999"}, hoverinfo="none"
    )

    nodes = list(graph.nodes)
    node_x = [layout[n][0] for n in nodes]
    node_y = [layout[n][1] for n in nodes]
    node_risk = [
        float(risk[n]) if 0 <= n < len(risk) and np.isfinite(risk[n]) else 0.0 for n in nodes
    ]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker={
            "color": node_risk,
            "colorscale": "RdYlGn_r",
            "size": 8,
            "showscale": True,
            "colorbar": {"title": "P(non-compliant)"},
        },
        text=[f"firm {n}: risk={r:.2f}" for n, r in zip(nodes, node_risk, strict=True)],
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Supply-network non-compliance diffusion map",
        showlegend=False,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


INTERACTIVE_FUNCS: dict[str, Any] = {
    "trajectory_fans": trajectory_fans_figure,
    "latent_pca": latent_pca_figure,
    "network_diffusion": network_diffusion_figure,
}


def write_figure(fig: Any, path: Path) -> Path:
    """Write a Plotly figure to a self-contained HTML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def make_all_interactive(cfg: SimWorldConfig) -> list[Path]:
    """Build every available interactive figure into ``reports/figures/*.html``."""
    fig_dir = Path(cfg.paths.reports) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, func in INTERACTIVE_FUNCS.items():
        try:
            fig = func(cfg)
        except Exception:
            log.exception("interactive figure %s raised; skipping", name)
            continue
        if fig is not None:
            written.append(write_figure(fig, fig_dir / f"{name}.html"))
    return written
