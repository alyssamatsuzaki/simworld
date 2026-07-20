"""Focused invariants for the shared pure decision rules."""

from __future__ import annotations

import numpy as np

from regworld.rules import SegmentAttributes, allocate_spend


def test_allocate_spend_excludes_dead_firms_and_conserves_budgets() -> None:
    segments = SegmentAttributes(
        weight=np.array([0.4, 0.6]),
        privacy=np.array([0.2, 0.8]),
        budget=np.array([4.0, 6.0]),
        trust0=np.array([0.5, 0.5]),
    )
    utility = np.array([[0.0, 4.0, 1.0], [1.0, 3.0, 0.0]])
    alive = np.array([True, False, True])
    market_mask = np.ones((2, 3), dtype=bool)

    spend, revenue, consumer_surplus = allocate_spend(utility, alive, market_mask, segments)

    np.testing.assert_allclose(spend.sum(axis=1), segments.budget)
    np.testing.assert_allclose(revenue, spend.sum(axis=0))
    assert np.all(spend[:, 1] == 0.0)
    assert revenue[1] == 0.0
    assert np.isfinite(consumer_surplus)
