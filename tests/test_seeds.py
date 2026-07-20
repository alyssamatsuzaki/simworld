"""Same seed → identical streams; different seeds differ (§13)."""

from __future__ import annotations

import numpy as np

from regworld.seeding import seed_everything, spawn


def test_same_seed_identical() -> None:
    a = seed_everything(42).random(10)
    b = seed_everything(42).random(10)
    assert np.array_equal(a, b)


def test_different_seed_differs() -> None:
    a = seed_everything(0).random(10)
    b = seed_everything(1).random(10)
    assert not np.array_equal(a, b)


def test_torch_seeded() -> None:
    import torch

    seed_everything(7)
    a = torch.rand(5)
    seed_everything(7)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_spawn_streams_independent_and_reproducible() -> None:
    kids1 = spawn(seed_everything(3), 4)
    kids2 = spawn(seed_everything(3), 4)
    for k1, k2 in zip(kids1, kids2, strict=True):
        assert np.array_equal(k1.random(5), k2.random(5))
    vals = [k.random(5) for k in spawn(seed_everything(3), 4)]
    assert not np.array_equal(vals[0], vals[1])
