"""GraphRSSM composite: encoder + macro RSSM + micro recurrence + heads.

Three arches share this class (§11 ablation #11): ``rssm_gnn`` (the full model),
``rssm_flat`` (RSSM, no message passing), ``gru_baseline`` (deterministic GRU on
aggregates, no latents, no graph). Macro stochasticity only —
``stochastic_level="node"`` is a declared ablation, not implemented in Phase 5.

Imagination-time micro approximation: firm dynamic features are frozen at their
last observed values except compliance, which is replaced each step by the node
head's own prediction; segment trust stays frozen (its aggregate is decoded by
the macro head). This keeps the graph in the imagination loop at the cost of one
GNN forward per imagined step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from einops import rearrange
from torch import nn

from regworld.models.encoder import ObsEncoder
from regworld.models.gnn import GraphTemplate
from regworld.models.heads import (
    AggregateHead,
    ContinueHead,
    NodeComplianceHead,
    RewardHead,
)
from regworld.models.rssm import MacroRSSM, RssmState
from regworld.training.losses import (
    kl_balanced,
    symlog_mse,
    two_hot_bins,
    two_hot_decode,
    two_hot_loss,
)

FIRM_DYNAMIC_DIM = 4  # compliant, alive, margin, cost_share (§8 node contract)
SEGMENT_DYNAMIC_DIM = 1  # trust


@dataclass
class ModelState:
    """Recurrent state carried by ``EmulatorEnv`` between steps."""

    core: RssmState | torch.Tensor  # RssmState, or GRU hidden for gru_baseline
    node_hidden: torch.Tensor | None  # (B * N_firm, H) micro GRU state
    firm_dynamic: torch.Tensor  # (B, N_firm, FIRM_DYNAMIC_DIM)
    segment_dynamic: torch.Tensor  # (B, N_segment, SEGMENT_DYNAMIC_DIM)


@dataclass(frozen=True)
class Decoded:
    """One imagined step, decoded."""

    aggregates: torch.Tensor  # (B, A) in natural units (symexp applied)
    node_probs: torch.Tensor  # (B, N_firm)
    reward: torch.Tensor  # (B,)
    continue_prob: torch.Tensor  # (B,)


class WorldModel(nn.Module):
    def __init__(
        self,
        *,
        arch: str,
        static_features: dict[str, torch.Tensor],
        aggregate_dim: int,
        action_dim: int,
        deter_dim: int,
        hidden_dim: int,
        latent_categories: int,
        latent_classes: int,
        gnn_layers: int,
        template: GraphTemplate,
        kl_free: float = 1.0,
        kl_balance: float = 0.8,
        stochastic_level: str = "macro",
    ) -> None:
        super().__init__()
        if stochastic_level != "macro":
            raise NotImplementedError("stochastic_level='node' is a declared ablation")
        self.arch = arch
        self.aggregate_dim = aggregate_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.kl_free = kl_free
        self.kl_balance = kl_balance
        for ntype, feats in static_features.items():
            self.register_buffer(f"static_{ntype}", feats.float())
        input_dims = {
            "firm": static_features["firm"].shape[1] + FIRM_DYNAMIC_DIM,
            "segment": static_features["segment"].shape[1] + SEGMENT_DYNAMIC_DIM,
            "association": static_features["association"].shape[1],
            "regulator": static_features["regulator"].shape[1],
        }
        self.encoder = ObsEncoder(
            arch=arch,
            input_dims=input_dims,
            aggregate_dim=aggregate_dim,
            hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
            template=template,
        )
        if arch == "gru_baseline":
            self.core_cell = nn.GRUCell(hidden_dim + action_dim, deter_dim)
            feature_dim = deter_dim
        else:
            self.rssm = MacroRSSM(
                action_dim=action_dim,
                embed_dim=hidden_dim,
                deter_dim=deter_dim,
                hidden_dim=hidden_dim,
                categories=latent_categories,
                classes=latent_classes,
            )
            feature_dim = self.rssm.feature_dim
        self.feature_dim = feature_dim
        self.context = nn.Sequential(nn.Linear(feature_dim + action_dim, hidden_dim), nn.SiLU())
        if arch == "rssm_gnn":
            self.node_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        else:
            # flat / gru_baseline node path: static projection + macro context
            self.firm_static_proj = nn.Linear(static_features["firm"].shape[1], hidden_dim)
        self.aggregate_head = AggregateHead(feature_dim, hidden_dim, aggregate_dim)
        self.node_head = NodeComplianceHead(hidden_dim, hidden_dim, hidden_dim)
        self.reward_head = RewardHead(feature_dim, hidden_dim)
        self.continue_head = ContinueHead(feature_dim, hidden_dim)
        self.register_buffer("reward_bins", two_hot_bins(self.reward_head.n_bins))

    # ---------------------------------------------------------------- features
    def _buffer(self, name: str) -> torch.Tensor:
        """Typed access to a registered buffer (``nn.Module.__getattr__`` unions)."""
        return cast(torch.Tensor, getattr(self, name))

    def _node_features(
        self, firm_dynamic: torch.Tensor, segment_dynamic: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """(M, N, F_dyn) dynamics + broadcast statics -> encoder feature dict."""
        m = firm_dynamic.shape[0]

        def stat(ntype: str) -> torch.Tensor:
            return self._buffer(f"static_{ntype}").unsqueeze(0).expand(m, -1, -1)

        return {
            "firm": torch.cat([stat("firm"), firm_dynamic], dim=-1),
            "segment": torch.cat([stat("segment"), segment_dynamic], dim=-1),
            "association": stat("association"),
            "regulator": stat("regulator"),
        }

    def _firm_static_embedding(self, batch: int) -> torch.Tensor:
        projected: torch.Tensor = self.firm_static_proj(self._buffer("static_firm"))
        return projected.unsqueeze(0).expand(batch, -1, -1)

    # ---------------------------------------------------------------- training
    def observe_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Teacher-forced losses over a (B, T, ...) sequence batch.

        ``action[:, t]`` produced observation ``t``; predictions for ``t >= 1``
        are graded (t = 0 is the conditioning frame).
        """
        firm, segment = batch["firm"], batch["segment"]
        aggregates, actions = batch["aggregate"], batch["action"]
        b, t = firm.shape[0], firm.shape[1]
        features = self._node_features(
            rearrange(firm, "b t n f -> (b t) n f"),
            rearrange(segment, "b t n f -> (b t) n f"),
        )
        embed_flat, firm_emb_flat = self.encoder(
            features, rearrange(aggregates, "b t a -> (b t) a")
        )
        embeds = rearrange(embed_flat, "(b t) e -> b t e", b=b)

        losses: dict[str, torch.Tensor] = {}
        if self.arch == "gru_baseline":
            hidden = torch.zeros(b, self.core_cell.hidden_size, device=firm.device)
            feats = []
            for step in range(t):
                hidden = self.core_cell(
                    torch.cat([embeds[:, step], actions[:, step]], dim=-1), hidden
                )
                feats.append(hidden)
            feature_seq = torch.stack(feats, dim=1)
            losses["kl"] = torch.zeros((), device=firm.device)
        else:
            posterior_states, priors, posteriors = self.rssm.observe(embeds, actions)
            feature_seq = torch.stack([s.feature for s in posterior_states], dim=1)
            losses["kl"] = kl_balanced(
                posteriors[:, 1:], priors[:, 1:], free=self.kl_free, balance=self.kl_balance
            )

        context = self.context(torch.cat([feature_seq, actions], dim=-1))  # (B, T, H)
        agg_pred = self.aggregate_head(feature_seq)
        losses["aggregate"] = symlog_mse(agg_pred[:, 1:], aggregates[:, 1:])
        reward_logits = self.reward_head(feature_seq[:, 1:])
        losses["reward"] = two_hot_loss(
            reward_logits, batch["reward"][:, 1:], self._buffer("reward_bins")
        )
        cont_logits = self.continue_head(feature_seq[:, 1:])
        losses["continue"] = nn.functional.binary_cross_entropy_with_logits(
            cont_logits, batch["cont"][:, 1:]
        )

        node_logits = self._node_logits_sequence(firm_emb_flat, context, b, t)
        target_y = firm[:, 1:, :, 0]
        alive = firm[:, 1:, :, 1]
        node_bce = nn.functional.binary_cross_entropy_with_logits(
            node_logits, target_y, reduction="none"
        )
        losses["node"] = (node_bce * alive).sum() / alive.sum().clamp(min=1.0)
        return losses

    def _node_logits_sequence(
        self,
        firm_emb_flat: torch.Tensor | None,
        context: torch.Tensor,
        b: int,
        t: int,
    ) -> torch.Tensor:
        """Predict node compliance for steps 1..T-1 from micro state at t-1."""
        if self.arch == "rssm_gnn":
            assert firm_emb_flat is not None
            firm_emb = rearrange(firm_emb_flat, "(b t) n h -> b t n h", b=b)
            n_firms = firm_emb.shape[2]
            hidden = torch.zeros(b * n_firms, self.hidden_dim, device=context.device)
            logits = []
            for step in range(1, t):
                ctx_step = context[:, step].unsqueeze(1).expand(-1, n_firms, -1)
                ctx_nodes = ctx_step.reshape(-1, self.hidden_dim)
                gru_in = torch.cat(
                    [firm_emb[:, step - 1].reshape(-1, self.hidden_dim), ctx_nodes], dim=-1
                )
                hidden = self.node_cell(gru_in, hidden)
                node_state = hidden.view(b, n_firms, self.hidden_dim)
                logits.append(self.node_head(node_state, context[:, step]))
            return torch.stack(logits, dim=1)
        # flat / gru_baseline: static projection + macro context carries dynamics
        static_emb = self._firm_static_embedding(b)
        logits = [self.node_head(static_emb, context[:, step]) for step in range(1, t)]
        return torch.stack(logits, dim=1)

    def open_loop(
        self,
        batch: dict[str, torch.Tensor],
        *,
        burn_in: int,
        horizon: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Condition on ``burn_in`` frames, roll the prior; returns
        (symlog aggregate predictions (B, K, A), node logits (B, K, N), start)."""
        firm, segment = batch["firm"], batch["segment"]
        aggregates, actions = batch["aggregate"], batch["action"]
        t = firm.shape[1]
        start = min(max(burn_in - 1, 0), t - 2)
        k = min(horizon, t - 1 - start)
        prefix = {
            "firm": firm[:, : start + 1],
            "segment": segment[:, : start + 1],
            "aggregate": aggregates[:, : start + 1],
            "action": actions[:, : start + 1],
        }
        state = self._posterior_state(prefix)
        agg_preds: list[torch.Tensor] = []
        node_logit_steps: list[torch.Tensor] = []
        for step in range(1, k + 1):
            state, decoded_symlog, node_logits = self._imagine_step_raw(
                state, actions[:, start + step], generator
            )
            agg_preds.append(decoded_symlog)
            node_logit_steps.append(node_logits)
        return torch.stack(agg_preds, dim=1), torch.stack(node_logit_steps, dim=1), start

    def imagination_losses(
        self,
        batch: dict[str, torch.Tensor],
        *,
        burn_in: int,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Open-loop drift: condition on ``burn_in`` frames, roll the prior
        ``horizon`` steps, grade decoded aggregates and node compliance."""
        firm = batch["firm"]
        aggregates = batch["aggregate"]
        agg_pred, node_logits_seq, start = self.open_loop(batch, burn_in=burn_in, horizon=horizon)
        k = agg_pred.shape[1]
        target = aggregates[:, start + 1 : start + 1 + k]
        losses = {"imag_aggregate": symlog_mse(agg_pred, target)}
        target_y = firm[:, start + 1 : start + 1 + k, :, 0]
        alive = firm[:, start + 1 : start + 1 + k, :, 1]
        node_bce = nn.functional.binary_cross_entropy_with_logits(
            node_logits_seq, target_y, reduction="none"
        )
        losses["imag_node"] = (node_bce * alive).sum() / alive.sum().clamp(min=1.0)
        return losses

    # ------------------------------------------------------------- imagination
    def _posterior_state(self, prefix: dict[str, torch.Tensor]) -> ModelState:
        """Filter the prefix (teacher-forced) and return the last state."""
        firm, segment = prefix["firm"], prefix["segment"]
        aggregates, actions = prefix["aggregate"], prefix["action"]
        b, t = firm.shape[0], firm.shape[1]
        features = self._node_features(
            rearrange(firm, "b t n f -> (b t) n f"),
            rearrange(segment, "b t n f -> (b t) n f"),
        )
        embed_flat, firm_emb_flat = self.encoder(
            features, rearrange(aggregates, "b t a -> (b t) a")
        )
        embeds = rearrange(embed_flat, "(b t) e -> b t e", b=b)
        core: RssmState | torch.Tensor
        if self.arch == "gru_baseline":
            hidden = torch.zeros(b, self.core_cell.hidden_size, device=firm.device)
            for step in range(t):
                hidden = self.core_cell(
                    torch.cat([embeds[:, step], actions[:, step]], dim=-1), hidden
                )
            core = hidden
            feature_seq = None
        else:
            states, _, _ = self.rssm.observe(embeds, actions)
            core = states[-1]
            feature_seq = torch.stack([s.feature for s in states], dim=1)
        node_hidden: torch.Tensor | None = None
        if self.arch == "rssm_gnn":
            assert firm_emb_flat is not None and feature_seq is not None
            firm_emb = rearrange(firm_emb_flat, "(b t) n h -> b t n h", b=b)
            n_firms = firm_emb.shape[2]
            context = self.context(torch.cat([feature_seq, actions], dim=-1))
            node_hidden = torch.zeros(b * n_firms, self.hidden_dim, device=firm.device)
            for step in range(1, t):
                ctx_step = context[:, step].unsqueeze(1).expand(-1, n_firms, -1)
                ctx_nodes = ctx_step.reshape(-1, self.hidden_dim)
                gru_in = torch.cat(
                    [firm_emb[:, step - 1].reshape(-1, self.hidden_dim), ctx_nodes], dim=-1
                )
                node_hidden = self.node_cell(gru_in, node_hidden)
        return ModelState(
            core=core,
            node_hidden=node_hidden,
            firm_dynamic=firm[:, -1].clone(),
            segment_dynamic=segment[:, -1].clone(),
        )

    def _imagine_step_raw(
        self,
        state: ModelState,
        action: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[ModelState, torch.Tensor, torch.Tensor]:
        """Advance one step; returns (state, symlog aggregate prediction, node logits)."""
        b = state.firm_dynamic.shape[0]
        core: RssmState | torch.Tensor
        feature: torch.Tensor
        if self.arch == "gru_baseline":
            assert isinstance(state.core, torch.Tensor)
            agg_prev = self.aggregate_head(state.core)
            features = self._node_features(state.firm_dynamic, state.segment_dynamic)
            embed, _ = self.encoder(features, agg_prev)
            hidden: torch.Tensor = self.core_cell(torch.cat([embed, action], dim=-1), state.core)
            core = hidden
            feature = hidden
        else:
            assert isinstance(state.core, RssmState)
            core, _ = self.rssm.img_step(state.core, action, generator)
            feature = core.feature
        context = self.context(torch.cat([feature, action], dim=-1))
        node_hidden = state.node_hidden
        if self.arch == "rssm_gnn":
            features = self._node_features(state.firm_dynamic, state.segment_dynamic)
            _, firm_emb = self.encoder(features, symexp_clamped(self.aggregate_head(feature)))
            assert firm_emb is not None and node_hidden is not None
            n_firms = firm_emb.shape[1]
            ctx_nodes = context.unsqueeze(1).expand(-1, n_firms, -1).reshape(-1, self.hidden_dim)
            gru_in = torch.cat([firm_emb.reshape(-1, self.hidden_dim), ctx_nodes], dim=-1)
            node_hidden = self.node_cell(gru_in, node_hidden)
            node_state = node_hidden.view(b, n_firms, self.hidden_dim)
        else:
            node_state = self._firm_static_embedding(b)
        node_logits = self.node_head(node_state, context)
        firm_dynamic = state.firm_dynamic.clone()
        firm_dynamic[:, :, 0] = torch.sigmoid(node_logits).detach()
        new_state = ModelState(
            core=core,
            node_hidden=node_hidden,
            firm_dynamic=firm_dynamic,
            segment_dynamic=state.segment_dynamic,
        )
        return new_state, self.aggregate_head(feature), node_logits

    @torch.no_grad()
    def initial_state(
        self,
        firm_dynamic: torch.Tensor,
        segment_dynamic: torch.Tensor,
        aggregates: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> ModelState:
        """Posterior state from a single conditioning frame (env reset)."""
        prefix = {
            "firm": firm_dynamic.unsqueeze(1),
            "segment": segment_dynamic.unsqueeze(1),
            "aggregate": aggregates.unsqueeze(1),
            "action": torch.zeros(
                firm_dynamic.shape[0], 1, self.action_dim, device=firm_dynamic.device
            ),
        }
        if generator is None:
            return self._posterior_state(prefix)
        # Re-run the single obs step with the caller's generator for determinism.
        features = self._node_features(firm_dynamic, segment_dynamic)
        embed, firm_emb = self.encoder(features, aggregates)
        if self.arch == "gru_baseline":
            hidden = torch.zeros(
                firm_dynamic.shape[0], self.core_cell.hidden_size, device=firm_dynamic.device
            )
            action0 = torch.zeros(firm_dynamic.shape[0], self.action_dim, device=embed.device)
            core: RssmState | torch.Tensor = self.core_cell(
                torch.cat([embed, action0], dim=-1), hidden
            )
        else:
            action0 = torch.zeros(firm_dynamic.shape[0], self.action_dim, device=embed.device)
            state0 = self.rssm.initial(firm_dynamic.shape[0], embed.device)
            core, _, _ = self.rssm.obs_step(state0, action0, embed, generator)
        node_hidden = None
        if self.arch == "rssm_gnn":
            assert firm_emb is not None
            node_hidden = torch.zeros(
                firm_dynamic.shape[0] * firm_emb.shape[1], self.hidden_dim, device=embed.device
            )
        return ModelState(
            core=core,
            node_hidden=node_hidden,
            firm_dynamic=firm_dynamic.clone(),
            segment_dynamic=segment_dynamic.clone(),
        )

    @torch.no_grad()
    def imagine_step(
        self,
        state: ModelState,
        action: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[ModelState, Decoded]:
        """One decoded env step (used by ``EmulatorEnv`` and the eval suite)."""
        new_state, agg_symlog, node_logits = self._imagine_step_raw(state, action, generator)
        feature = (
            new_state.core.feature if isinstance(new_state.core, RssmState) else new_state.core
        )
        decoded = Decoded(
            aggregates=symexp_clamped(agg_symlog),
            node_probs=torch.sigmoid(node_logits),
            reward=two_hot_decode(self.reward_head(feature), self._buffer("reward_bins")),
            continue_prob=torch.sigmoid(self.continue_head(feature)),
        )
        return new_state, decoded


def symexp_clamped(symlog_values: torch.Tensor, max_abs: float = 50_000.0) -> torch.Tensor:
    """Symexp with a hard clamp so an untrained head cannot emit infinities."""
    from regworld.training.losses import symexp

    return symexp(symlog_values).clamp(-max_abs, max_abs)


def build_world_model(
    cfg: Any,
    static_features: dict[str, torch.Tensor],
    template: GraphTemplate,
    *,
    aggregate_dim: int,
    action_dim: int = 4,
    arch: str | None = None,
) -> WorldModel:
    """Construct a WorldModel from ``cfg.emulator``; ``arch`` overrides for ablations."""
    em = cfg.emulator
    return WorldModel(
        arch=arch or em.arch,
        static_features=static_features,
        aggregate_dim=aggregate_dim,
        action_dim=action_dim,
        deter_dim=em.deter_dim,
        hidden_dim=em.hidden_dim,
        latent_categories=em.latent_categories,
        latent_classes=em.latent_classes,
        gnn_layers=em.gnn_layers,
        template=template,
        kl_free=em.kl_free,
        kl_balance=em.kl_balance,
        stochastic_level=em.stochastic_level,
    )
