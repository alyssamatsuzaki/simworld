"""Causal discovery on the observables, and why it is wrong (5e).

PC and GES assume faithfulness and no latent confounders. Both are violated here by
construction - ``z`` is latent - so the recovered graph will differ from the true
DAG. Reporting the structural Hamming distance and the reason inoculates the team
against trusting a discovery algorithm on confounded observational data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from regworld.causal.graph import OUTCOME, TREATMENT, true_dag_edges

# Observed variables handed to discovery (the latent capacity is, of course, absent).
_DISCOVERY_VARS = (
    "audited_prev",
    "outcome_reported",
    "perceived_risk",
    "cost_share",
    "neighbor_compliant_share",
    "size_decile",
    "data_intensity",
    "cost_index",
)

# Map DAG node names (graph.py) onto the observed columns discovery actually sees.
_NODE_TO_COLUMN = {
    TREATMENT: "audited_prev",
    OUTCOME: "outcome_reported",
    "perceived_risk": "perceived_risk",
    "cost_index": "cost_index",
    "data_intensity": "data_intensity",
    "size_decile": "size_decile",
    "neighbor_compliance": "neighbor_compliant_share",
}


@dataclass(frozen=True)
class DiscoveryReport:
    """Recovered-vs-true structural Hamming distance for PC and GES."""

    pc_shd: int
    ges_shd: int
    n_true_edges: int
    reason: str


def _true_skeleton_on_observed() -> set[frozenset[str]]:
    """Undirected true-DAG edges restricted to nodes discovery can see."""
    inv = {node: col for node, col in _NODE_TO_COLUMN.items()}
    edges: set[frozenset[str]] = set()
    for src, dst in true_dag_edges():
        if src in inv and dst in inv:
            edges.add(frozenset({inv[src], inv[dst]}))
    return edges


def _adjacency_to_skeleton(adj: np.ndarray, names: list[str]) -> set[frozenset[str]]:
    edges: set[frozenset[str]] = set()
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0 or adj[j, i] != 0:
                edges.add(frozenset({names[i], names[j]}))
    return edges


def _shd(recovered: set[frozenset[str]], truth: set[frozenset[str]]) -> int:
    """Skeleton structural Hamming distance: symmetric-difference edge count."""
    return len(recovered ^ truth)


def discover(panel: pl.DataFrame, *, seed: int = 0) -> DiscoveryReport:
    """Run PC and GES on the observed variables; score skeletons against the truth."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges

    names = [c for c in _DISCOVERY_VARS if c in panel.columns]
    data = np.column_stack([panel[c].to_numpy().astype(np.float64) for c in names])
    # Small deterministic jitter avoids degenerate perfectly-collinear columns.
    rng = np.random.default_rng(seed)
    data = data + rng.normal(0.0, 1e-6, size=data.shape)

    pc_graph = pc(data, alpha=0.05, indep_test="fisherz", show_progress=False)
    pc_skeleton = _adjacency_to_skeleton(pc_graph.G.graph, names)

    ges_result = ges(data, score_func="local_score_BIC")
    ges_skeleton = _adjacency_to_skeleton(ges_result["G"].graph, names)

    truth = _true_skeleton_on_observed()
    return DiscoveryReport(
        pc_shd=_shd(pc_skeleton, truth),
        ges_shd=_shd(ges_skeleton, truth),
        n_true_edges=len(truth),
        reason=(
            "faithfulness and causal sufficiency are violated by construction: the "
            "latent capacity z confounds compliance, so no observational discovery "
            "algorithm can recover the true DAG."
        ),
    )
