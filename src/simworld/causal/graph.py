"""The assumed causal DAG (§7.7), in two variants.

The *analyst's* DAG omits the latent capacity ``z`` - the analyst does not know it
exists - and therefore looks identifiable via a backdoor set of observed controls.
That is the trap. The *true* DAG adds ``z`` as an unobserved common driver of
compliance that the size proxy cannot fully block (size is observed only as a
coarse decile), so no backdoor set built from observables closes it.

The whole pipeline runs on the analyst's DAG; grading uses the true one.
"""

from __future__ import annotations

# Treatment / outcome shared by both DAGs. The effect of interest is the audit
# channel audited -> perceived_risk -> compliant_next (§7.7).
TREATMENT = "audited"
OUTCOME = "compliant_next"

# Observed covariates the analyst can condition on. `size_decile` is the coarsened
# stand-in for continuous size; `cost_index` is the noisy compliance-cost proxy.
_OBSERVED = (
    "size_decile",
    "cost_index",
    "data_intensity",
    "neighbor_compliance",
    "assoc_compliance",
    "compliant_lag",
)

# The single latent node the true DAG adds. Continuous size and the true cost are
# also unobserved, but `capacity` is the one that does the damage the demo needs.
LATENT = "capacity"


def _gml(edges: list[tuple[str, str]], nodes: list[str]) -> str:
    """Serialise a node/edge set as a GML string DoWhy can parse."""
    lines = ["graph [", "  directed 1"]
    for name in nodes:
        lines.append("  node [")
        lines.append(f'    id "{name}"')
        lines.append(f'    label "{name}"')
        lines.append("  ]")
    for src, dst in edges:
        lines.append("  edge [")
        lines.append(f'    source "{src}"')
        lines.append(f'    target "{dst}"')
        lines.append("  ]")
    lines.append("]")
    return "\n".join(lines)


def _shared_edges() -> list[tuple[str, str]]:
    """Edges present in both DAGs (everything except the capacity paths)."""
    return [
        ("size_decile", TREATMENT),  # regulator targets by size
        ("size_decile", "cost_index"),
        ("data_intensity", "cost_index"),
        ("cost_index", OUTCOME),
        (TREATMENT, "perceived_risk"),
        ("perceived_risk", OUTCOME),
        ("neighbor_compliance", OUTCOME),
        ("assoc_compliance", OUTCOME),
        ("compliant_lag", OUTCOME),
    ]


def analyst_dag() -> str:
    """The DAG the analyst believes: observed nodes only, looks identifiable."""
    nodes = [TREATMENT, OUTCOME, "perceived_risk", *_OBSERVED]
    return _gml(_shared_edges(), sorted(set(nodes)))


def true_dag() -> str:
    """The DAG known by construction: adds the unobserved capacity confounder.

    ``size_decile -> capacity -> compliant_next`` looks blockable by conditioning on
    ``size_decile``, but the analyst only sees the coarse decile while the DGP wires
    ``capacity`` to *continuous* size, so conditioning on the decile leaves residual
    open backdoor flow. Encoding it as its own latent node makes ``identify_effect``
    on this DAG (with ``capacity`` unobserved) refuse the observed-only backdoor set.
    """
    edges = [*_shared_edges(), ("size_decile", LATENT), (LATENT, OUTCOME)]
    nodes = [TREATMENT, OUTCOME, "perceived_risk", LATENT, *_OBSERVED]
    return _gml(edges, sorted(set(nodes)))


def observed_adjustment_set() -> list[str]:
    """The backdoor controls an analyst would use (everything observed but the pair)."""
    return list(_OBSERVED)


def true_dag_edges() -> list[tuple[str, str]]:
    """Directed edges of the true DAG, for structural-Hamming-distance scoring."""
    return [*_shared_edges(), ("size_decile", LATENT), (LATENT, OUTCOME)]
