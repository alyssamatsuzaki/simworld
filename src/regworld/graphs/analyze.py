"""Graph metrics (§10 Stage 2): degree distributions, clustering, communities,
centrality, cascade reachability — and assortativity-by-z, the cheap sanity check
that the homophily knob is doing something (≈0 wellspecified, >0.2 confounded).

`z` is ground-truth-only: pass it only from world-build code or evaluation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from regworld.graphs.build import RegGraphs


def _powerlaw_alpha(degrees: np.ndarray, k_min: int = 2) -> float:
    """Continuous MLE exponent over the degree tail (Clauset-style, fixed k_min)."""
    tail = degrees[degrees >= k_min].astype(np.float64)
    if tail.size < 10:
        return float("nan")
    return float(1.0 + tail.size / np.sum(np.log(tail / (k_min - 0.5))))


def analyze_graphs(
    reg: RegGraphs, *, observed: bool, z: np.ndarray | None = None, top_k: int = 10
) -> dict[str, Any]:
    supply = reg.supply_obs if observed else reg.supply_true
    influence = reg.influence_obs if observed else reg.influence_true
    und = supply.to_undirected()
    degrees = np.array([d for _, d in und.degree()])
    metrics: dict[str, Any] = {
        "observed": observed,
        "supply_nodes": supply.number_of_nodes(),
        "supply_edges": supply.number_of_edges(),
        "supply_self_loops": int(nx.number_of_selfloops(supply)),
        "supply_weakly_connected": bool(
            nx.is_weakly_connected(supply) if supply.number_of_edges() else False
        ),
        "supply_mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "supply_powerlaw_alpha": _powerlaw_alpha(degrees),
        "supply_clustering": float(nx.average_clustering(und)),
        "influence_clustering": float(nx.average_clustering(influence)),
    }
    if nx.is_connected(influence):
        metrics["influence_avg_path_length"] = float(nx.average_shortest_path_length(influence))
    else:  # pragma: no cover - WS with these params is connected in practice
        big = influence.subgraph(max(nx.connected_components(influence), key=len))
        metrics["influence_avg_path_length"] = float(nx.average_shortest_path_length(big))
    communities = nx.community.louvain_communities(und, seed=0)
    metrics["supply_n_communities"] = len(communities)
    # the degraded observed graph can be disconnected; eigenvector centrality is
    # only defined per component, so compute it on the largest one
    if und.number_of_edges():
        giant = und.subgraph(max(nx.connected_components(und), key=len))
        eig = nx.eigenvector_centrality_numpy(giant)
    else:
        eig = {}
    btw = nx.betweenness_centrality(und, k=min(64, und.number_of_nodes()), seed=0)
    metrics["top_eigenvector_firms"] = [
        int(n) for n, _ in sorted(eig.items(), key=lambda kv: -kv[1])[:top_k]
    ]
    metrics["top_betweenness_firms"] = [
        int(n) for n, _ in sorted(btw.items(), key=lambda kv: -kv[1])[:top_k]
    ]
    # cascade reachability: how much of the graph the highest-degree firm can reach
    if und.number_of_nodes():
        hub = int(max(dict(und.degree()).items(), key=lambda kv: kv[1])[0])
        reach = len(nx.descendants(supply.reverse(copy=False), hub))
        metrics["hub_upstream_reach_frac"] = reach / max(supply.number_of_nodes(), 1)
    if z is not None:
        for node in und.nodes():
            und.nodes[node]["z"] = float(z[int(node)])
        metrics["assortativity_z"] = float(
            nx.numeric_assortativity_coefficient(und, "z")
            if und.number_of_edges()
            else float("nan")
        )
    return metrics
