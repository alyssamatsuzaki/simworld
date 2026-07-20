"""DreamerV3 loss ingredients: symlog, two-hot, KL balancing with free bits.

The exact recipe the plan names (§10 Stages 6+7): reconstruction + KL with
0.8/0.2 balancing toward the prior, free bits at 1.0 nat, symlog on continuous
targets, two-hot reward. Without these you tune per-config and hate your life.
"""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def symlog_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in symlog space; the prediction is already in symlog space."""
    return torch.mean((prediction - symlog(target)) ** 2)


def two_hot_bins(
    n_bins: int, low: float = -3.0, high: float = 3.0, device: torch.device | None = None
) -> torch.Tensor:
    """Symlog-spaced bin centers; symexp maps them back to reward units."""
    return torch.linspace(low, high, n_bins, device=device)


def two_hot_encode(value: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode symlog(value) as weight split across its two nearest bins."""
    target = symlog(value).clamp(bins[0], bins[-1])
    idx_hi = torch.searchsorted(bins, target, right=True).clamp(1, bins.numel() - 1)
    idx_lo = idx_hi - 1
    width = bins[idx_hi] - bins[idx_lo]
    w_hi = ((target - bins[idx_lo]) / width).clamp(0.0, 1.0)
    encoding = torch.zeros(*value.shape, bins.numel(), device=value.device)
    encoding.scatter_(-1, idx_lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
    encoding.scatter_add_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return encoding


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value under the categorical, mapped back through symexp."""
    return symexp(torch.sum(torch.softmax(logits, dim=-1) * bins, dim=-1))


def two_hot_loss(logits: torch.Tensor, value: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    target = two_hot_encode(value, bins)
    return -torch.mean(torch.sum(target * torch.log_softmax(logits, dim=-1), dim=-1))


def _kl_categorical(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for (..., cats, classes) log-probs, summed over categories."""
    return torch.sum(p_log.exp() * (p_log - q_log), dim=(-2, -1))


def kl_balanced(
    posterior_log: torch.Tensor,
    prior_log: torch.Tensor,
    *,
    free: float = 1.0,
    balance: float = 0.8,
) -> torch.Tensor:
    """KL balancing (DreamerV3): train the prior toward the posterior harder than
    the posterior toward the prior, each term floored at ``free`` nats."""
    dynamics = _kl_categorical(posterior_log.detach(), prior_log).mean()
    representation = _kl_categorical(posterior_log, prior_log.detach()).mean()
    free_t = torch.tensor(free, device=posterior_log.device)
    return balance * torch.maximum(dynamics, free_t) + (1.0 - balance) * torch.maximum(
        representation, free_t
    )
