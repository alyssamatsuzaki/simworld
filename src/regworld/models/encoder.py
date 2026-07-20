"""Observation encoder ``g_t``: pooled micro state + hand-built aggregates.

Feeding the §7.6 aggregate vector alongside the pooled GNN embedding is not
cheating — it is telling the model what the analyst already knows. The three
arches share this module: ``rssm_gnn`` pools HeteroGNN firm embeddings
(mean + max + learned attention), ``rssm_flat`` mean-pools raw node features
(no message passing), ``gru_baseline`` sees the aggregates alone.
"""

from __future__ import annotations

import torch
from torch import nn

from regworld.models.gnn import GraphTemplate, MicroGNN


class ObsEncoder(nn.Module):
    """Per-quarter encoder; returns ``(g_t, firm_embeddings | None)``."""

    def __init__(
        self,
        *,
        arch: str,
        input_dims: dict[str, int],
        aggregate_dim: int,
        hidden_dim: int,
        gnn_layers: int,
        template: GraphTemplate,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.embed_dim = hidden_dim
        self.gnn: MicroGNN | None = None
        if arch == "rssm_gnn":
            self.gnn = MicroGNN(input_dims, hidden_dim, gnn_layers, template)
            self.attention = nn.Linear(hidden_dim, 1)
            # firm mean + max + attention pool, plus the mean embedding of every
            # other node type (keeps isolated types like the regulator in the
            # gradient path and in the macro context).
            head_in = 6 * hidden_dim + aggregate_dim
        elif arch == "rssm_flat":
            head_in = input_dims["firm"] + input_dims["segment"] + aggregate_dim
        elif arch == "gru_baseline":
            head_in = aggregate_dim
        else:
            raise ValueError(f"unknown emulator arch: {arch!r}")
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, features: dict[str, torch.Tensor], aggregates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.arch == "gru_baseline":
            return self.head(aggregates), None
        if self.arch == "rssm_flat":
            pooled = torch.cat(
                [
                    features["firm"].mean(dim=1),
                    features["segment"].mean(dim=1),
                    aggregates,
                ],
                dim=-1,
            )
            return self.head(pooled), None
        assert self.gnn is not None
        embeddings = self.gnn(features)
        firm = embeddings["firm"]  # (M, N_f, H)
        weights = torch.softmax(self.attention(firm), dim=1)
        pooled = torch.cat(
            [
                firm.mean(dim=1),
                firm.max(dim=1).values,
                (weights * firm).sum(dim=1),
                embeddings["segment"].mean(dim=1),
                embeddings["association"].mean(dim=1),
                embeddings["regulator"].mean(dim=1),
                aggregates,
            ],
            dim=-1,
        )
        return self.head(pooled), firm
