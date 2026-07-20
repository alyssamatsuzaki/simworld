"""Stage 6+7 acceptance: forward + imagination shapes, gradient flow, no NaN.

Runs on a synthetic miniature world (no artifacts needed) so it stays in the
fast suite: a hand-built heterogeneous graph, random dynamics, and tiny latent
dimensions. All three arches are exercised; the full-size model differs only in
widths.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from regworld.models.gnn import GraphTemplate
from regworld.models.rssm import MacroRSSM
from regworld.models.world_model import ModelState, WorldModel

N_FIRMS, N_SEGMENTS, N_ASSOC = 12, 3, 2
AGG_DIM, ACTION_DIM = 10, 4
BATCH, STEPS = 2, 6
ARCHES = ("rssm_gnn", "rssm_flat", "gru_baseline")


def tiny_template(rng: np.random.Generator) -> GraphTemplate:
    def pairs(n_src: int, n_dst: int, n_edges: int) -> torch.Tensor:
        src = rng.integers(0, n_src, n_edges)
        dst = rng.integers(0, n_dst, n_edges)
        return torch.tensor(np.stack([src, dst]), dtype=torch.long)

    supply = pairs(N_FIRMS, N_FIRMS, 20)
    market = pairs(N_SEGMENTS, N_FIRMS, 15)
    member = pairs(N_FIRMS, N_ASSOC, N_FIRMS)
    return GraphTemplate(
        {"firm": N_FIRMS, "segment": N_SEGMENTS, "association": N_ASSOC, "regulator": 1},
        {
            ("firm", "supplies", "firm"): supply,
            ("firm", "supplied_by", "firm"): supply.flip(0),
            ("segment", "influences", "segment"): pairs(N_SEGMENTS, N_SEGMENTS, 6),
            ("segment", "buys_from", "firm"): market,
            ("firm", "sells_to", "segment"): market.flip(0),
            ("firm", "member_of", "association"): member,
            ("association", "has_member", "firm"): member.flip(0),
        },
    )


def tiny_model(arch: str, seed: int = 0) -> WorldModel:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    static = {
        "firm": torch.randn(N_FIRMS, 5),
        "segment": torch.randn(N_SEGMENTS, 2),
        "association": torch.ones(N_ASSOC, 1),
        "regulator": torch.ones(1, 1),
    }
    return WorldModel(
        arch=arch,
        static_features=static,
        aggregate_dim=AGG_DIM,
        action_dim=ACTION_DIM,
        deter_dim=16,
        hidden_dim=32,
        latent_categories=8,
        latent_classes=8,
        gnn_layers=2,
        template=tiny_template(rng),
    )


def tiny_batch(seed: int = 0, *, deterministic_targets: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    firm = torch.rand(BATCH, STEPS, N_FIRMS, 4)
    firm[..., 0] = (
        (torch.arange(N_FIRMS) % 2).float()
        if deterministic_targets
        else (firm[..., 0] > 0.5).float()
    )
    firm[..., 1] = 1.0
    batch = {
        "firm": firm,
        "segment": torch.rand(BATCH, STEPS, N_SEGMENTS, 1),
        "aggregate": torch.rand(BATCH, STEPS, AGG_DIM),
        "action": torch.rand(BATCH, STEPS, ACTION_DIM),
        "reward": torch.zeros(BATCH, STEPS),
        "cont": torch.ones(BATCH, STEPS),
    }
    return batch


@pytest.mark.parametrize("arch", ARCHES)
def test_observe_losses_finite(arch: str) -> None:
    model = tiny_model(arch)
    losses = model.observe_losses(tiny_batch())
    for name, value in losses.items():
        assert value.ndim == 0, name
        assert torch.isfinite(value), f"{name} is not finite"
    if arch == "gru_baseline":
        assert float(losses["kl"]) == 0.0


@pytest.mark.parametrize("arch", ARCHES)
def test_imagination_losses_finite(arch: str) -> None:
    model = tiny_model(arch)
    losses = model.imagination_losses(tiny_batch(), burn_in=2, horizon=3)
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} is not finite"


@pytest.mark.parametrize("arch", ARCHES)
def test_gradients_reach_every_parameter(arch: str) -> None:
    model = tiny_model(arch)
    batch = tiny_batch()
    losses = model.observe_losses(batch)
    losses.update(model.imagination_losses(batch, burn_in=2, horizon=3))
    torch.stack(list(losses.values())).sum().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert missing == [], f"no/NaN gradient for: {missing}"


def test_rssm_shapes_and_imagination() -> None:
    rssm = MacroRSSM(
        action_dim=ACTION_DIM,
        embed_dim=32,
        deter_dim=16,
        hidden_dim=32,
        categories=8,
        classes=8,
    )
    embeds = torch.randn(BATCH, STEPS, 32)
    actions = torch.randn(BATCH, STEPS, ACTION_DIM)
    states, priors, posteriors = rssm.observe(embeds, actions)
    assert len(states) == STEPS
    assert priors.shape == (BATCH, STEPS, 8, 8)
    assert posteriors.shape == (BATCH, STEPS, 8, 8)
    assert states[-1].feature.shape == (BATCH, 16 + 64)
    # one-hot straight-through: each category sums to one
    stoch = states[-1].stoch.view(BATCH, 8, 8)
    assert torch.allclose(stoch.detach().sum(-1), torch.ones(BATCH, 8))
    imagined = rssm.imagine(states[-1], torch.randn(BATCH, 4, ACTION_DIM))
    assert len(imagined) == 4
    assert torch.isfinite(imagined[-1].feature).all()


@pytest.mark.parametrize("arch", ARCHES)
def test_env_step_interface(arch: str) -> None:
    """initial_state -> imagine_step round trip with a seeded generator."""
    model = tiny_model(arch)
    model.eval()
    firm = torch.rand(1, N_FIRMS, 4)
    segment = torch.rand(1, N_SEGMENTS, 1)
    aggregates = torch.rand(1, AGG_DIM)
    generator = torch.Generator().manual_seed(3)
    state = model.initial_state(firm, segment, aggregates, generator)
    assert isinstance(state, ModelState)
    action = torch.rand(1, ACTION_DIM)
    new_state, decoded = model.imagine_step(state, action, generator)
    assert decoded.aggregates.shape == (1, AGG_DIM)
    assert decoded.node_probs.shape == (1, N_FIRMS)
    assert decoded.reward.shape == (1,)
    assert decoded.continue_prob.shape == (1,)
    assert torch.isfinite(decoded.aggregates).all()
    assert ((decoded.node_probs >= 0) & (decoded.node_probs <= 1)).all()
    # the imagined compliance belief is written back into the firm features
    assert torch.allclose(new_state.firm_dynamic[:, :, 0], decoded.node_probs)


def test_determinism_with_seeded_generator() -> None:
    model = tiny_model("rssm_gnn")
    model.eval()
    firm = torch.rand(1, N_FIRMS, 4)
    segment = torch.rand(1, N_SEGMENTS, 1)
    aggregates = torch.rand(1, AGG_DIM)
    action = torch.rand(1, ACTION_DIM)
    outputs = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(11)
        state = model.initial_state(firm, segment, aggregates, generator)
        _, decoded = model.imagine_step(state, action, generator)
        outputs.append(decoded.aggregates)
    assert torch.equal(outputs[0], outputs[1])
