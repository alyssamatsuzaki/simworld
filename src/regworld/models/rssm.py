"""Macro RSSM: deterministic GRU core + 32x32 categorical stochastic latents.

DreamerV2 discrete latents with straight-through gradients and a 1% uniform mix
(unimix) so no class collapses to exactly zero probability; prior ``p(z_t | d_t)``
and posterior ``q(z_t | d_t, g_t)`` where ``g_t`` is the encoded observation.
Macro dynamics carry all injected stochasticity; the micro path is conditionally
deterministic given ``[d_t, z_t]`` (the factorization in §10 Stages 6+7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn


@dataclass(frozen=True)
class RssmState:
    """One time-slice of the macro latent: deterministic + sampled stochastic."""

    deter: torch.Tensor  # (B, deter_dim)
    stoch: torch.Tensor  # (B, categories * classes) straight-through one-hots

    @property
    def feature(self) -> torch.Tensor:
        """The decoder input ``[d_t, z_t]``."""
        return torch.cat([self.deter, self.stoch], dim=-1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class MacroRSSM(nn.Module):
    """Recurrent state-space model over the aggregate (macro) system state."""

    def __init__(
        self,
        *,
        action_dim: int,
        embed_dim: int,
        deter_dim: int,
        hidden_dim: int,
        categories: int,
        classes: int,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.categories = categories
        self.classes = classes
        self.deter_dim = deter_dim
        self.stoch_dim = categories * classes
        self.unimix = unimix
        self.cell = nn.GRUCell(self.stoch_dim + action_dim, deter_dim)
        self.prior_net = _mlp(deter_dim, hidden_dim, self.stoch_dim)
        self.posterior_net = _mlp(deter_dim + embed_dim, hidden_dim, self.stoch_dim)

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_dim

    def initial(self, batch: int, device: torch.device) -> RssmState:
        return RssmState(
            deter=torch.zeros(batch, self.deter_dim, device=device),
            stoch=torch.zeros(batch, self.stoch_dim, device=device),
        )

    def _log_probs(self, raw_logits: torch.Tensor) -> torch.Tensor:
        """(B, cats*classes) raw output -> (B, cats, classes) unimixed log-probs."""
        logits = rearrange(
            raw_logits, "b (cat cls) -> b cat cls", cat=self.categories, cls=self.classes
        )
        probs = torch.softmax(logits, dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.classes
        return torch.log(probs)

    def sample(
        self, log_probs: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Straight-through one-hot sample, flattened to (B, cats*classes)."""
        probs = log_probs.exp()
        flat = rearrange(probs, "b cat cls -> (b cat) cls")
        index = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        one_hot = torch.zeros_like(flat).scatter_(-1, index.unsqueeze(-1), 1.0)
        one_hot = rearrange(one_hot, "(b cat) cls -> b cat cls", cat=self.categories)
        sample = one_hot + probs - probs.detach()
        flat_sample: torch.Tensor = rearrange(sample, "b cat cls -> b (cat cls)")
        return flat_sample

    def _advance(self, state: RssmState, action: torch.Tensor) -> torch.Tensor:
        deter: torch.Tensor = self.cell(torch.cat([state.stoch, action], dim=-1), state.deter)
        return deter

    def obs_step(
        self,
        state: RssmState,
        action: torch.Tensor,
        embed: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[RssmState, torch.Tensor, torch.Tensor]:
        """One posterior step; returns (state, prior log-probs, posterior log-probs)."""
        deter = self._advance(state, action)
        prior = self._log_probs(self.prior_net(deter))
        posterior = self._log_probs(self.posterior_net(torch.cat([deter, embed], dim=-1)))
        stoch = self.sample(posterior, generator)
        return RssmState(deter=deter, stoch=stoch), prior, posterior

    def img_step(
        self,
        state: RssmState,
        action: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[RssmState, torch.Tensor]:
        """One prior-only (imagination) step; returns (state, prior log-probs)."""
        deter = self._advance(state, action)
        prior = self._log_probs(self.prior_net(deter))
        stoch = self.sample(prior, generator)
        return RssmState(deter=deter, stoch=stoch), prior

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        state: RssmState | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[list[RssmState], torch.Tensor, torch.Tensor]:
        """Teacher-forced filtering over (B, T, ...) sequences.

        ``actions[:, t]`` is the action that produced observation ``embeds[:, t]``.
        Returns per-step states plus stacked (B, T, cats, classes) prior and
        posterior log-probs.
        """
        batch, steps = embeds.shape[0], embeds.shape[1]
        if state is None:
            state = self.initial(batch, embeds.device)
        states: list[RssmState] = []
        priors: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        for t in range(steps):
            state, prior, posterior = self.obs_step(state, actions[:, t], embeds[:, t], generator)
            states.append(state)
            priors.append(prior)
            posteriors.append(posterior)
        return states, torch.stack(priors, dim=1), torch.stack(posteriors, dim=1)

    def imagine(
        self,
        state: RssmState,
        actions: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> list[RssmState]:
        """Open-loop prior rollout under an action sequence (B, K, action_dim)."""
        states: list[RssmState] = []
        for k in range(actions.shape[1]):
            state, _ = self.img_step(state, actions[:, k], generator)
            states.append(state)
        return states
