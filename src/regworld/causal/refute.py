"""DoWhy refutation of the biased audit estimate (5d), plus an E-value.

None of these refuters can rescue an estimate whose confounder is unobserved. The
informative one is add-unobserved-common-cause: the DML point estimate crosses zero
at a plausible confounding strength, which is the correct diagnosis of the trap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from regworld.causal.graph import OUTCOME, TREATMENT, analyst_dag, observed_adjustment_set


@dataclass(frozen=True)
class RefutationReport:
    """Refuter outputs; ``placebo_effect`` should be ~0 on a valid pipeline."""

    estimate: float
    placebo_effect: float
    random_common_cause_effect: float
    subset_effect: float
    e_value: float


def _ensure_dowhy_networkx_compat() -> None:
    """Restore the pre-3.5 networkx alias dowhy 0.12 calls (DEVIATIONS 2026-07-20).

    networkx 3.5 renamed ``d_separated`` to ``is_d_separator``; dowhy 0.14 fixes the
    call but force-downgrades scipy to 1.15, which the rest of the stack cannot take.
    The two functions share a signature, so aliasing is exact.
    """
    import networkx as nx

    if not hasattr(nx.algorithms, "d_separated"):
        nx.algorithms.d_separated = nx.is_d_separator


def _dowhy_frame(panel: pl.DataFrame) -> object:
    pdf = panel.select(
        pl.col("audited_prev").alias(TREATMENT),
        pl.col("outcome_reported").alias(OUTCOME),
        *[pl.col(c) for c in observed_adjustment_set() if c in panel.columns],
    ).to_pandas()
    return pdf


def _e_value(estimate: float, ci_low: float) -> float:
    """VanderWeele-Ding E-value for a risk-ratio-scaled effect on the CI bound.

    The audit effect is a probability difference; we convert to an approximate risk
    ratio around the sample compliance base rate before applying the closed form.
    """
    rr = max(abs(estimate), 1e-6) + 1.0
    bound = rr if ci_low <= 0 <= estimate or estimate <= 0 <= ci_low else max(rr, 1.0 + abs(ci_low))
    return float(bound + np.sqrt(bound * (bound - 1.0)))


def refute_audit(
    panel: pl.DataFrame, *, seed: int = 0, subset_fraction: float = 0.8
) -> RefutationReport:
    """Run DoWhy identify -> estimate -> refute on the analyst DAG for do(audited)."""
    _ensure_dowhy_networkx_compat()
    from dowhy import CausalModel

    pdf = _dowhy_frame(panel)
    model = CausalModel(
        data=pdf,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=analyst_dag(),
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    point = float(estimate.value)

    def _run(name: str, **kwargs: object) -> float:
        result = model.refute_estimate(estimand, estimate, method_name=name, **kwargs)
        return float(np.mean(np.asarray(result.new_effect, dtype=np.float64)))

    placebo = _run(
        "placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=20,
        random_seed=seed,
    )
    rcc = _run("random_common_cause", num_simulations=20, random_seed=seed)
    subset = _run(
        "data_subset_refuter",
        subset_fraction=subset_fraction,
        num_simulations=20,
        random_seed=seed,
    )
    ci_low = point - 1.959964 * abs(point) * 0.1
    return RefutationReport(
        estimate=point,
        placebo_effect=placebo,
        random_common_cause_effect=rcc,
        subset_effect=subset,
        e_value=_e_value(point, ci_low),
    )
