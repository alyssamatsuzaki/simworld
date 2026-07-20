"""Stage 2 acceptance tests (§10): structure, homophily knob, PyG round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from regworld.graphs.analyze import analyze_graphs
from regworld.graphs.build import RegGraphs, build_graphs
from regworld.types import RegWorldConfig, validate_config

from .conftest import compose_cfg


def _world_inputs(cfg: RegWorldConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, s = cfg.population.n_firms, cfg.population.n_consumer_segments
    size = rng.lognormal(0, 1.1, n)
    size = size / np.median(size)
    logs = np.log(size)
    logs_std = (logs - logs.mean()) / logs.std()
    rho = cfg.dgp.corr_z_size
    z = rho * logs_std + np.sqrt(max(1 - rho**2, 0)) * rng.normal(0, 1, n)
    return {
        "size": size,
        "sector": rng.integers(0, cfg.population.n_sectors, n),
        "z": z,
        "association": rng.integers(-1, cfg.population.n_associations, n),
        "seg_pref": rng.dirichlet(np.full(cfg.population.n_sectors, 2.0), size=s),
    }


@pytest.fixture(scope="module")
def confounded() -> tuple[RegWorldConfig, RegGraphs, np.ndarray]:
    cfg = validate_config(compose_cfg("profile=smoke", "dgp=confounded"))
    inputs = _world_inputs(cfg)
    rng = np.random.default_rng(1)
    return cfg, build_graphs(cfg, rng, **inputs), inputs["z"]


def test_no_self_loops_and_connected(confounded: tuple) -> None:
    _, reg, _ = confounded
    m = analyze_graphs(reg, observed=False)
    assert m["supply_self_loops"] == 0
    assert m["supply_weakly_connected"]


def test_scale_free_exponent_in_sane_band(confounded: tuple) -> None:
    _, reg, _ = confounded
    m = analyze_graphs(reg, observed=False)
    assert 1.3 < m["supply_powerlaw_alpha"] < 5.0


def test_ws_small_world_properties(confounded: tuple) -> None:
    _, reg, _ = confounded
    m = analyze_graphs(reg, observed=False)
    assert m["influence_clustering"] > 0.2  # high clustering
    assert m["influence_avg_path_length"] < 4.0  # short paths at smoke scale


def test_assortativity_by_z_tracks_homophily_knob() -> None:
    """~0 under wellspecified, clearly positive under confounded (§7.2)."""
    cfg_w = validate_config(compose_cfg("profile=smoke", "dgp=wellspecified"))
    cfg_c = validate_config(compose_cfg("profile=smoke", "dgp=confounded"))
    assert cfg_w.network.homophily == 0.0 and cfg_c.network.homophily == 1.5
    inputs_w = _world_inputs(cfg_w, seed=3)
    inputs_c = _world_inputs(cfg_c, seed=3)
    g_w = build_graphs(cfg_w, np.random.default_rng(4), **inputs_w)
    g_c = build_graphs(cfg_c, np.random.default_rng(4), **inputs_c)
    a_w = analyze_graphs(g_w, observed=False, z=inputs_w["z"])["assortativity_z"]
    a_c = analyze_graphs(g_c, observed=False, z=inputs_c["z"])["assortativity_z"]
    assert abs(a_w) < 0.12, f"wellspecified assortativity should be ~0, got {a_w}"
    assert a_c > 0.2, f"confounded assortativity should exceed 0.2, got {a_c}"


def test_observed_graph_is_degraded_not_destroyed(confounded: tuple) -> None:
    _cfg, reg, _ = confounded
    n_true = reg.supply_true.number_of_edges()
    n_obs = reg.supply_obs.number_of_edges()
    # ~20% dropped, ~3% spurious -> observed within (0.7, 0.95) of true
    assert 0.65 * n_true < n_obs < 0.95 * n_true


def test_market_covers_every_firm_and_keeps_preferential_extras(confounded: tuple) -> None:
    cfg, reg, _ = confounded
    firm_nodes = [f"firm_{i}" for i in range(cfg.population.n_firms)]
    assert all(reg.market.degree(node) > 0 for node in firm_nodes)
    # One mandatory coverage edge per firm, plus preferential extra links for
    # every segment. At smoke scale there are enough non-neighbours for all extras.
    expected_edges = cfg.population.n_firms + (
        cfg.population.n_consumer_segments * cfg.network.firms_per_segment
    )
    assert reg.market.number_of_edges() == expected_edges
    assert all(
        reg.market.degree(f"seg_{j}") >= cfg.network.firms_per_segment
        for j in range(cfg.population.n_consumer_segments)
    )


def test_static_feature_contract_excludes_dynamic_state(confounded: tuple) -> None:
    import polars as pl

    from regworld.graphs.to_pyg import (
        node_feature_contract,
        static_feature_shapes,
        static_node_features,
    )

    cfg, _, _ = confounded
    n = cfg.population.n_firms
    registry = pl.DataFrame(
        {
            "firm_id": np.arange(n),
            "sector": np.arange(n) % cfg.population.n_sectors,
            "size_decile": np.arange(n) % 10,
            "data_intensity": np.linspace(0.0, 1.0, n),
            "cost_index": np.linspace(0.0, 2.0, n),
        }
    )
    survey = pl.DataFrame(
        {
            "segment_id": [0, 0, 2],
            "privacy_bucket": [2, 2, 0],
        }
    )
    contract = node_feature_contract(cfg)
    features = static_node_features(cfg, registry, survey)
    assert {node_type: values.shape for node_type, values in features.items()} == (
        static_feature_shapes(cfg)
    )
    assert contract["firm"].static_dim == cfg.population.n_sectors + 3
    assert contract["firm"].dynamic == ("compliant", "alive", "margin", "cost_share")
    assert contract["regulator"].static == ("bias",)
    assert "budget_used" not in contract["regulator"].static
    # Segment 1 was not observed: neutral imputation is explicit in the flag.
    assert features["segment"][1].tolist() == [0.5, 0.0]


def test_pyg_round_trip_and_heteroconv_forward(confounded: tuple) -> None:
    import polars as pl
    from torch_geometric.nn import HeteroConv, SAGEConv

    from regworld.graphs.build import edges_frame
    from regworld.graphs.to_pyg import hetero_from_edges

    cfg, reg, _ = confounded
    n, s = cfg.population.n_firms, cfg.population.n_consumer_segments

    def df(pairs: list) -> pl.DataFrame:
        return pl.DataFrame({"src": [a for a, _ in pairs], "dst": [b for _, b in pairs]}).cast(
            {"src": pl.Utf8, "dst": pl.Utf8}
        )

    edges = {
        "supply_edges": df(edges_frame(reg.supply_obs)),
        "influence_edges": df(edges_frame(reg.influence_obs)),
        "market_edges": df(edges_frame(reg.market)),
        "membership_edges": df(edges_frame(reg.membership)),
    }
    feats = {
        "firm": np.random.default_rng(0).random((n, 4)).astype(np.float32),
        "segment": np.random.default_rng(1).random((s, 2)).astype(np.float32),
        "association": np.zeros((cfg.population.n_associations, 1), dtype=np.float32),
        "regulator": np.ones((1, 1), dtype=np.float32),
    }
    data = hetero_from_edges(cfg, edges, feats)
    # round trip: node and edge counts preserved
    assert data["firm"].x.shape[0] == n
    assert data["segment"].x.shape[0] == s
    assert data["firm", "supplies", "firm"].edge_index.shape[1] == (
        reg.supply_obs.number_of_edges()
    )
    assert data["segment", "buys_from", "firm"].edge_index.shape[1] == (
        reg.market.number_of_edges()
    )
    assert ("regulator", "audits", "firm") not in data.edge_types
    assert ("firm", "audited_by", "regulator") not in data.edge_types
    conv = HeteroConv(
        {
            ("firm", "supplies", "firm"): SAGEConv(4, 8),
            ("firm", "supplied_by", "firm"): SAGEConv(4, 8),
            ("segment", "buys_from", "firm"): SAGEConv((2, 4), 8),
            ("segment", "influences", "segment"): SAGEConv(2, 8),
        },
        aggr="sum",
    )
    out = conv(data.x_dict, data.edge_index_dict)
    assert out["firm"].shape == (n, 8)
    assert out["segment"].shape == (s, 8)
    assert torch.isfinite(out["firm"]).all()


def test_pyg_rejects_market_with_uncovered_firm(confounded: tuple) -> None:
    import polars as pl

    from regworld.graphs.build import edges_frame
    from regworld.graphs.to_pyg import hetero_from_edges

    cfg, reg, _ = confounded

    def df(pairs: list) -> pl.DataFrame:
        return pl.DataFrame({"src": [a for a, _ in pairs], "dst": [b for _, b in pairs]}).cast(
            {"src": pl.Utf8, "dst": pl.Utf8}
        )

    market_without_firm_zero = [pair for pair in edges_frame(reg.market) if "firm_0" not in pair]
    edges = {
        "supply_edges": df(edges_frame(reg.supply_obs)),
        "influence_edges": df(edges_frame(reg.influence_obs)),
        "market_edges": df(market_without_firm_zero),
        "membership_edges": df(edges_frame(reg.membership)),
    }
    feats = {
        "firm": np.ones((cfg.population.n_firms, 1), dtype=np.float32),
        "segment": np.ones((cfg.population.n_consumer_segments, 1), dtype=np.float32),
        "association": np.ones((cfg.population.n_associations, 1), dtype=np.float32),
        "regulator": np.ones((1, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="market_edges must cover every firm"):
        hetero_from_edges(cfg, edges, feats)
