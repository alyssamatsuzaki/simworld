"""Observed graph edges + observed features → PyG HeteroData (§10 Stage 2).

Node features are observed attributes only. The four static relations get reverse
edges where their directions differ. Audit events and other quarter-varying state
belong to Phase 5 and are deliberately absent from this static artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from regworld.types import RegWorldConfig


@dataclass(frozen=True)
class NodeFeatureContract:
    """Names of static Stage 2 inputs and dynamic Phase 5 extensions."""

    static: tuple[str, ...]
    dynamic: tuple[str, ...]

    @property
    def static_dim(self) -> int:
        return len(self.static)

    @property
    def dynamic_dim(self) -> int:
        return len(self.dynamic)


def node_feature_contract(cfg: RegWorldConfig) -> dict[str, NodeFeatureContract]:
    """Return the non-leaky feature contract for every heterogeneous node type.

    Static tensors contain only fields available from the observed registry and
    survey. A bias column keeps featureless institution nodes representable without
    pretending that their future publicity, budget, or policy state is known.
    """
    sector_names = tuple(f"sector_{sector}" for sector in range(cfg.population.n_sectors))
    return {
        "firm": NodeFeatureContract(
            static=("size_decile_norm", "data_intensity", "cost_index_norm", *sector_names),
            dynamic=("compliant", "alive", "margin", "cost_share"),
        ),
        "segment": NodeFeatureContract(
            static=("privacy_bucket_norm_imputed", "privacy_bucket_observed"),
            dynamic=("trust",),
        ),
        "association": NodeFeatureContract(static=("bias",), dynamic=("publicity",)),
        "regulator": NodeFeatureContract(
            static=("bias",),
            dynamic=("budget_used", "targeting", "phase_progress"),
        ),
    }


def static_feature_shapes(cfg: RegWorldConfig) -> dict[str, tuple[int, int]]:
    """Expected Stage 2 feature shapes, separated from dynamic Phase 5 state."""
    contract = node_feature_contract(cfg)
    return {
        "firm": (cfg.population.n_firms, contract["firm"].static_dim),
        "segment": (cfg.population.n_consumer_segments, contract["segment"].static_dim),
        "association": (
            cfg.population.n_associations,
            contract["association"].static_dim,
        ),
        "regulator": (1, contract["regulator"].static_dim),
    }


def hetero_from_edges(
    cfg: RegWorldConfig,
    edges: dict[str, pl.DataFrame],
    node_features: dict[str, np.ndarray],
) -> Any:  # HeteroData; typed Any so torch_geometric stays a lazy import
    """Build HeteroData from observed edge lists (§8 graphs/) and feature blocks.

    `edges` keys: supply_edges, influence_edges, market_edges, membership_edges.
    `node_features` keys: firm, segment, association, regulator.
    """
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for ntype in ("firm", "segment", "association", "regulator"):
        feats = node_features[ntype]
        data[ntype].x = torch.as_tensor(feats, dtype=torch.float32)

    def _pairs(df: pl.DataFrame, strip_src: str = "", strip_dst: str = "") -> torch.Tensor:
        if df.height == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        src = [int(s.removeprefix(strip_src)) for s in df["src"].to_list()]
        dst = [int(s.removeprefix(strip_dst)) for s in df["dst"].to_list()]
        return torch.tensor([src, dst], dtype=torch.long)

    supply = _pairs(edges["supply_edges"])
    data["firm", "supplies", "firm"].edge_index = supply
    data["firm", "supplied_by", "firm"].edge_index = supply.flip(0)

    infl = _pairs(edges["influence_edges"])
    infl_sym = torch.cat([infl, infl.flip(0)], dim=1)
    data["segment", "influences", "segment"].edge_index = infl_sym

    # market/membership edge lists carry "seg_j"/"firm_i"/"assoc_a" string ids
    mdf = edges["market_edges"]
    seg_side = mdf["src"].str.starts_with("seg_")
    seg_ids = (
        pl.concat([mdf.filter(seg_side)["src"], mdf.filter(~seg_side)["dst"]])
        .str.strip_prefix("seg_")
        .cast(pl.Int64)
        .to_list()
    )
    firm_ids = (
        pl.concat([mdf.filter(seg_side)["dst"], mdf.filter(~seg_side)["src"]])
        .str.strip_prefix("firm_")
        .cast(pl.Int64)
        .to_list()
    )
    market = torch.tensor([seg_ids, firm_ids], dtype=torch.long)
    missing_firms = sorted(set(range(node_features["firm"].shape[0])) - set(firm_ids))
    if missing_firms:
        preview = ", ".join(str(i) for i in missing_firms[:10])
        raise ValueError(
            "market_edges must cover every firm; "
            f"missing {len(missing_firms)} firm(s), starting with: {preview}"
        )
    data["segment", "buys_from", "firm"].edge_index = market
    data["firm", "sells_to", "segment"].edge_index = market.flip(0)

    bdf = edges["membership_edges"]
    firm_side = bdf["src"].str.starts_with("firm_")
    m_firms = (
        pl.concat([bdf.filter(firm_side)["src"], bdf.filter(~firm_side)["dst"]])
        .str.strip_prefix("firm_")
        .cast(pl.Int64)
        .to_list()
    )
    m_assoc = (
        pl.concat([bdf.filter(firm_side)["dst"], bdf.filter(~firm_side)["src"]])
        .str.strip_prefix("assoc_")
        .cast(pl.Int64)
        .to_list()
    )
    member = torch.tensor([m_firms, m_assoc], dtype=torch.long)
    data["firm", "member_of", "association"].edge_index = member
    data["association", "has_member", "firm"].edge_index = member.flip(0)

    # The regulator is intentionally isolated in the static graph. Actual audits
    # are dynamic event edges and are attached per quarter in Phase 5.
    return data


def static_node_features(
    cfg: RegWorldConfig, registry: pl.DataFrame, survey: pl.DataFrame
) -> dict[str, np.ndarray]:
    """Build observed static features matching :func:`static_feature_shapes`.

    Values unavailable in the observed corpus are not synthesized as current
    state. Missing privacy buckets use neutral imputation plus an observed flag;
    time-varying blocks are added by Phase 5.
    """
    n_sectors = cfg.population.n_sectors
    reg_sorted = registry.sort("firm_id")
    sector = reg_sorted["sector"].to_numpy()
    one_hot = np.eye(n_sectors)[sector]
    firm = np.column_stack(
        [
            reg_sorted["size_decile"].to_numpy() / 9.0,
            reg_sorted["data_intensity"].to_numpy(),
            np.clip(reg_sorted["cost_index"].to_numpy(), 0, 5) / 5.0,
            one_hot,
        ]
    )
    s = cfg.population.n_consumer_segments
    privacy = np.full(s, 0.5)
    privacy_observed = np.zeros(s)
    if survey.height:
        med = survey.group_by("segment_id").agg(pl.col("privacy_bucket").median())
        for row in med.iter_rows():
            privacy[int(row[0])] = row[1] / 2.0
            privacy_observed[int(row[0])] = 1.0
    segment = np.column_stack([privacy, privacy_observed])
    association = np.ones((cfg.population.n_associations, 1))
    regulator = np.ones((1, 1))
    features = {
        "firm": firm,
        "segment": segment,
        "association": association,
        "regulator": regulator,
    }
    expected = static_feature_shapes(cfg)
    actual = {node_type: values.shape for node_type, values in features.items()}
    if actual != expected:  # pragma: no cover - guards future schema edits
        raise ValueError(f"static feature shape mismatch: expected {expected}, got {actual}")
    return features
