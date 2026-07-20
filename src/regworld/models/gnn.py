"""Micro path: heterogeneous message passing over the observed static graph.

The graph is fixed across episodes and quarters (Stage 2's ``hetero_observed.pt``),
so a batch of (episode, quarter) slices is one forward over ``M`` stacked copies
of the same graph with index offsets — no per-sample Batch construction. The
regulator node receives no static edges (audit events are macro context here),
so every layer carries a per-type residual projection to keep isolated node
types updating.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from einops import rearrange
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

NODE_TYPES = ("firm", "segment", "association", "regulator")
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("firm", "supplies", "firm"),
    ("firm", "supplied_by", "firm"),
    ("segment", "influences", "segment"),
    ("segment", "buys_from", "firm"),
    ("firm", "sells_to", "segment"),
    ("firm", "member_of", "association"),
    ("association", "has_member", "firm"),
)


class GraphTemplate:
    """Node counts + base edge indices of the observed graph, with M-copy cache."""

    def __init__(
        self,
        node_counts: dict[str, int],
        edge_index: dict[tuple[str, str, str], torch.Tensor],
    ) -> None:
        self.node_counts = node_counts
        self.edge_index = {k: v.long() for k, v in edge_index.items()}
        self._cache: dict[int, dict[tuple[str, str, str], torch.Tensor]] = {}

    @classmethod
    def from_hetero_data(cls, data: Any) -> GraphTemplate:
        counts = {ntype: int(data[ntype].x.shape[0]) for ntype in data.node_types}
        edges = {
            etype: data[etype].edge_index
            for etype in data.edge_types
            if data[etype].edge_index.shape[1] > 0
        }
        return cls(counts, edges)

    def replicated(self, copies: int) -> dict[tuple[str, str, str], torch.Tensor]:
        """Edge indices for ``copies`` stacked graphs, cached per copy count."""
        if copies not in self._cache:
            out: dict[tuple[str, str, str], torch.Tensor] = {}
            for (src_t, rel, dst_t), base in self.edge_index.items():
                n_edges = base.shape[1]
                src_off = (
                    torch.arange(copies, device=base.device) * self.node_counts[src_t]
                ).repeat_interleave(n_edges)
                dst_off = (
                    torch.arange(copies, device=base.device) * self.node_counts[dst_t]
                ).repeat_interleave(n_edges)
                tiled = base.repeat(1, copies)
                out[(src_t, rel, dst_t)] = torch.stack([tiled[0] + src_off, tiled[1] + dst_off])
            self._cache[copies] = out
        return self._cache[copies]


class MicroGNN(nn.Module):
    """Projection + ``gnn_layers`` rounds of HeteroConv(SAGE) with type residuals."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int,
        n_layers: int,
        template: GraphTemplate,
    ) -> None:
        super().__init__()
        self.template = template
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict(
            {ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in NODE_TYPES}
        )
        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        present = tuple(template.edge_index)
        for _ in range(n_layers):
            self.convs.append(
                HeteroConv(
                    {
                        etype: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                        for etype in EDGE_TYPES
                        if etype in present
                    },
                    aggr="sum",
                )
            )
            self.residuals.append(
                nn.ModuleDict({ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in NODE_TYPES})
            )
        self.act = nn.SiLU()

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``features[ntype]`` is (M, N_ntype, F_ntype); returns (M, N_ntype, H)."""
        copies = features["firm"].shape[0]
        x = {
            ntype: self.act(self.projections[ntype](rearrange(feat, "m n f -> (m n) f")))
            for ntype, feat in features.items()
        }
        edge_index = self.template.replicated(copies)
        for conv, residual_module in zip(self.convs, self.residuals, strict=True):
            residual = cast(nn.ModuleDict, residual_module)
            messages = conv(x, edge_index)
            x = {
                ntype: self.act(residual[ntype](x[ntype]) + messages.get(ntype, 0.0)) for ntype in x
            }
        return {
            ntype: cast(torch.Tensor, rearrange(h, "(m n) h -> m n h", m=copies))
            for ntype, h in x.items()
        }
