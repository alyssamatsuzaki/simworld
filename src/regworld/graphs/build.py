"""NetworkX construction of the interaction structure (§7.2).

Two versions of every graph exist and both matter: the TRUE graph (used by the DGP)
and the OBSERVED graph (20% of supply/influence edges missing, 3% spurious) used by
calibration and the emulator. The gap between them is a result, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse

from regworld.rules import Graphs
from regworld.types import RegWorldConfig


@dataclass(frozen=True)
class RegGraphs:
    """The NetworkX bundle: true and observed pairs plus node index maps."""

    supply_true: nx.DiGraph  # (firm, supplies, firm), edge i -> j: i supplies j
    supply_obs: nx.DiGraph
    influence_true: nx.Graph  # (segment, influences, segment)
    influence_obs: nx.Graph
    market: nx.Graph  # (segment, buys_from, firm), bipartite; observed exactly
    membership: nx.Graph  # (firm, member_of, association); registry data, exact

    def runtime(self, *, observed: bool, n_firms: int, n_segments: int) -> Graphs:
        """Adjacency bundle the dynamics run on (§7.4)."""
        sup = self.supply_obs if observed else self.supply_true
        inf = self.influence_obs if observed else self.influence_true
        a_dir = nx.to_scipy_sparse_array(sup, nodelist=range(n_firms), format="csr")
        a_und = ((a_dir + a_dir.T) > 0).astype(np.float64)
        inf_a = nx.to_scipy_sparse_array(inf, nodelist=range(n_segments), format="csr")
        inf_a = inf_a.astype(np.float64)
        row_sums = np.asarray(inf_a.sum(axis=1)).ravel()
        inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        inf_norm = sparse.diags(inv) @ inf_a
        market_mask = np.zeros((n_segments, n_firms), dtype=bool)
        for s, f in self.market.edges():
            seg, firm = (s, f) if str(s).startswith("seg") else (f, s)
            market_mask[int(str(seg)[4:]), int(str(firm)[5:])] = True
        return Graphs(
            supply_und=sparse.csr_matrix(a_und),
            influence=sparse.csr_matrix(inf_norm),
            market_mask=market_mask,
        )


def _supply_graph(
    size: np.ndarray,
    sector: np.ndarray,
    z: np.ndarray,
    m: int,
    alpha: float,
    homophily: float,
    rng: np.random.Generator,
    smallworld: bool = False,
) -> nx.DiGraph:
    """Preferential attachment (m=2) directed by size rank, with sector and capacity
    homophily: P(i -> j) ~ deg(j)^alpha * exp(-lambda_homoph * |z_i - z_j|) (§7.2).

    lambda_homoph is THE knob that confounds peer-effect estimates.
    """
    n = size.shape[0]
    order = np.argsort(-size)  # largest firm first: bigger firms are earlier, better-connected
    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(range(n))
    if smallworld:  # network=smallworld ablation: WS ring over the size ordering
        ws = nx.watts_strogatz_graph(n, k=max(2, 2 * m), p=0.1, seed=int(rng.integers(2**31)))
        for a, b in ws.edges():
            i, j = order[a], order[b]
            g.add_edge(int(i), int(j))
        return g
    deg = np.ones(n)  # +1 smoothing so early nodes are reachable
    placed: list[int] = []
    for idx, i in enumerate(order):
        if idx == 0:
            placed.append(int(i))
            continue
        cand = np.array(placed)
        w = deg[cand] ** alpha * np.exp(-homophily * np.abs(z[i] - z[cand]))
        w = w * np.where(sector[cand] == sector[i], 1.5, 1.0)  # sector homophily
        w = w / w.sum()
        k = min(m, cand.size)
        targets = rng.choice(cand, size=k, replace=False, p=w)
        for j in targets:
            g.add_edge(int(i), int(j))  # i supplies j (j is the larger/earlier customer)
            deg[j] += 1
            deg[i] += 1
        placed.append(int(i))
    return g


def _degrade_directed(
    g: nx.DiGraph, drop: float, spurious: float, n: int, rng: np.random.Generator
) -> nx.DiGraph:
    obs: nx.DiGraph = nx.DiGraph()
    obs.add_nodes_from(range(n))
    edges = list(g.edges())
    keep = rng.random(len(edges)) >= drop
    for (a, b), k in zip(edges, keep, strict=True):
        if k:
            obs.add_edge(a, b)
    n_spurious = int(spurious * len(edges))
    for _ in range(n_spurious):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b and not g.has_edge(a, b):
            obs.add_edge(a, b)
    return obs


def _degrade_undirected(
    g: nx.Graph, drop: float, spurious: float, n: int, rng: np.random.Generator
) -> nx.Graph:
    obs: nx.Graph = nx.Graph()
    obs.add_nodes_from(range(n))
    edges = list(g.edges())
    keep = rng.random(len(edges)) >= drop
    for (a, b), k in zip(edges, keep, strict=True):
        if k:
            obs.add_edge(a, b)
    for _ in range(int(spurious * len(edges))):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b and not g.has_edge(a, b):
            obs.add_edge(a, b)
    return obs


def build_graphs(
    cfg: RegWorldConfig,
    rng: np.random.Generator,
    *,
    size: np.ndarray,
    sector: np.ndarray,
    z: np.ndarray,
    association: np.ndarray,
    seg_pref: np.ndarray,
) -> RegGraphs:
    """The five generators of §7.2, seeded. `seg_pref` is (S, K) sector preference."""
    n = cfg.population.n_firms
    s = cfg.population.n_consumer_segments
    net = cfg.network

    supply_true = _supply_graph(
        size,
        sector,
        z,
        net.supply_m,
        net.alpha,
        net.homophily,
        rng,
        smallworld=net.name == "smallworld",
    )
    supply_obs = _degrade_directed(supply_true, cfg.dgp.edge_dropout, cfg.dgp.edge_spurious, n, rng)

    influence_true = nx.watts_strogatz_graph(
        s, k=min(net.ws_k, s - 1 - (s % 2 == 0)), p=net.ws_p, seed=int(rng.integers(2**31))
    )
    influence_obs = _degrade_undirected(
        influence_true, cfg.dgp.edge_dropout, cfg.dgp.edge_spurious, s, rng
    )

    market: nx.Graph = nx.Graph()
    market.add_nodes_from([f"seg_{j}" for j in range(s)], bipartite=0)
    market.add_nodes_from([f"firm_{i}" for i in range(n)], bipartite=1)

    # A market edge is the only route through which a firm can receive demand.
    # Give every firm a coverage edge first so no firm starts structurally unable
    # to trade. Segment choice still follows the firm's sector match.
    for i in range(n):
        w_segment = seg_pref[:, sector[i]].astype(np.float64, copy=True)
        w_segment = w_segment / w_segment.sum()
        j = int(rng.choice(s, p=w_segment))
        market.add_edge(f"seg_{j}", f"firm_{i}")

    # Retain the preferential market structure as extra links. Sampling only
    # non-neighbours makes `firms_per_segment` mean additional opportunities,
    # rather than silently spending draws on coverage edges already present.
    for j in range(s):
        connected = {int(str(node)[5:]) for node in market.neighbors(f"seg_{j}")}
        candidates = np.array([i for i in range(n) if i not in connected], dtype=np.int64)
        if candidates.size == 0:
            continue
        w = size[candidates] * seg_pref[j, sector[candidates]]
        w = w / w.sum()
        k = min(net.firms_per_segment, candidates.size)
        firms_j = rng.choice(candidates, size=k, replace=False, p=w)
        for i in firms_j:
            market.add_edge(f"seg_{j}", f"firm_{int(i)}")

    membership: nx.Graph = nx.Graph()
    membership.add_nodes_from([f"assoc_{a}" for a in range(cfg.population.n_associations)])
    membership.add_nodes_from([f"firm_{i}" for i in range(n)])
    for i in range(n):
        if association[i] >= 0:
            membership.add_edge(f"firm_{i}", f"assoc_{int(association[i])}")

    return RegGraphs(
        supply_true=supply_true,
        supply_obs=supply_obs,
        influence_true=influence_true,
        influence_obs=influence_obs,
        market=market,
        membership=membership,
    )


def edges_frame(g: nx.Graph | nx.DiGraph) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in g.edges()]


def supply_from_edges(edges: np.ndarray, n: int) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(a), int(b)) for a, b in edges)
    return g
