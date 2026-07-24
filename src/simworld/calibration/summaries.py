"""Low-dimensional macro summaries used by simulation-based calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

SUMMARY_NAMES = (
    "terminal_compliance",
    "time_to_half_compliance",
    "terminal_hhi",
    "mean_trust",
    "terminal_exit_rate",
    "adoption_inflection_quarter",
)


def _first_half(compliance: np.ndarray) -> float:
    reached = np.flatnonzero(compliance >= 0.5)
    return float(reached[0] + 1) if reached.size else float(compliance.size + 1)


def _inflection(compliance: np.ndarray) -> float:
    if compliance.size < 3:
        return float(max(compliance.size, 1))
    # The steepest adoption increment is a stable discrete definition of the
    # S-curve's inflection point and is meaningful even for a noisy curve.
    return float(np.argmax(np.diff(compliance)) + 2)


def summary_statistics(trajectory: Any) -> np.ndarray:
    """Return the six Stage-4b statistics from a trajectory or aggregate table.

    Supported inputs are a Polars aggregate-series frame, a NumPy ``(T, >=6)``
    outcome matrix, and Mesa/Torch trajectory objects exposing ``outcomes`` or
    ``outcome_matrix``. The returned order is fixed by :data:`SUMMARY_NAMES`.
    """
    if isinstance(trajectory, pl.DataFrame):
        compliance = trajectory["compliance_rate_obs"].to_numpy()
        hhi = trajectory["hhi_obs"].to_numpy()
        trust = trajectory["mean_trust_obs"].to_numpy()
        exit_rate = trajectory["exit_rate_obs"].to_numpy()
    elif isinstance(trajectory, np.ndarray):
        if trajectory.ndim != 2 or trajectory.shape[1] < 6:
            raise ValueError("outcome matrix must have shape (quarter, >=6)")
        compliance = trajectory[:, 0]
        hhi = trajectory[:, 2]
        trust = trajectory[:, 3]
        exit_rate = trajectory[:, 5]
    elif hasattr(trajectory, "outcome_matrix"):
        matrix = trajectory.outcome_matrix()
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
        return summary_statistics(np.asarray(matrix, dtype=np.float64))
    elif hasattr(trajectory, "outcomes"):
        outcomes = trajectory.outcomes
        matrix = np.asarray(
            [
                [
                    float(x.compliance_rate),
                    float(x.compliance_rate_weighted),
                    float(x.hhi),
                    float(x.mean_trust),
                    float(x.consumer_surplus),
                    float(x.exit_rate_cum),
                ]
                for x in outcomes
            ],
            dtype=np.float64,
        )
        return summary_statistics(matrix)
    else:
        raise TypeError(f"unsupported trajectory type: {type(trajectory)!r}")

    if compliance.size == 0:
        raise ValueError("cannot summarize an empty trajectory")
    values = np.asarray(
        [
            compliance[-1],
            _first_half(compliance),
            hhi[-1],
            np.mean(trust),
            exit_rate[-1],
            _inflection(compliance),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("macro summary contains non-finite values")
    return values
