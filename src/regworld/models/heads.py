"""Decoder heads (§10 Stages 6+7): aggregates, node compliance, reward, continue.

Aggregates train under symlog MSE; the reward head is two-hot over symlog-spaced
bins (both per DreamerV3); node compliance and the continuation flag are BCE.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class AggregateHead(nn.Module):
    """Predicts the §7.6-derived aggregate vector in symlog space."""

    def __init__(self, feature_dim: int, hidden_dim: int, aggregate_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, aggregate_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(feature)
        return out


class NodeComplianceHead(nn.Module):
    """Per-firm ``logit(y_{i,t})`` from the micro state and the macro context."""

    def __init__(self, node_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(node_dim + context_dim, hidden_dim, 1)

    def forward(self, node_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """``node_state`` (M, N, D), ``context`` (M, C) broadcast per node -> (M, N)."""
        expanded = context.unsqueeze(1).expand(-1, node_state.shape[1], -1)
        logits: torch.Tensor = self.net(torch.cat([node_state, expanded], dim=-1))
        return logits.squeeze(-1)


class RewardHead(nn.Module):
    """Two-hot categorical over symlog-spaced bins."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_bins: int = 63) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.net = _mlp(feature_dim, hidden_dim, n_bins)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(feature)
        return out


class ContinueHead(nn.Module):
    """Probability the episode has not hit systemic collapse."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        logit: torch.Tensor = self.net(feature)
        return logit.squeeze(-1)
