"""Stage 5a-5e: identify, estimate four ways, refute, discover — all on observed data."""

import json
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

from simworld.causal.did import estimate_did
from simworld.causal.estimate import cate_by_group, dml_audit, dml_onset, naive_logit_audit
from simworld.causal.graph import (
    OUTCOME,
    TREATMENT,
    analyst_dag,
    observed_adjustment_set,
    true_dag,
)
from simworld.causal.refute import refute_audit
from simworld.data.ingest import read_panel_analysis
from simworld.logging_conf import get_logger, setup_logging
from simworld.seeding import seed_everything
from simworld.types import validate_config

log = get_logger(__name__)


def _identifiability_report(panel: object) -> dict[str, object]:
    """PLAN 5b — 'report both': run identify_effect on the analyst AND true DAGs.

    Reports the raw DoWhy verdict on each graph plus the honest interpretation.
    The subtlety this DGP plants (§7.7): capacity ``z`` does *not* cause the
    treatment — the confounding runs ``size -> audited`` and ``size -> z ->
    compliant_next``, so conditioning on size symbolically closes the backdoor and
    DoWhy declares BOTH DAGs identifiable. That symbolic success is the real trap:
    the panel carries only the coarse ``size_decile``, a proxy for the continuous
    size the confounding actually flows through, so the "identified" estimand is
    still biased (the four-number gate and the E-value quantify how much).
    """
    from simworld.causal.refute import _dowhy_frame, _ensure_dowhy_networkx_compat

    _ensure_dowhy_networkx_compat()
    from dowhy import CausalModel

    pdf = _dowhy_frame(panel)  # type: ignore[arg-type]

    def _has_backdoor(graph_gml: str) -> bool:
        try:
            model = CausalModel(data=pdf, treatment=TREATMENT, outcome=OUTCOME, graph=graph_gml)
            estimand = model.identify_effect(proceed_when_unidentifiable=False)
            backdoor = getattr(estimand, "estimands", {}).get("backdoor")
            return bool(backdoor and backdoor.get("estimand") is not None)
        except Exception:  # dowhy raises when the effect is not identifiable
            return False

    return {
        "analyst_dag_identifiable": _has_backdoor(analyst_dag()),
        "true_dag_identifiable": _has_backdoor(true_dag()),
        "note": (
            "DoWhy conditions on size_decile and declares both DAGs identifiable — "
            "the trap. Structural identifiability does not imply unbiasedness here: "
            "size_decile is a coarsening of the continuous size the confounding runs "
            "through, so the identified audit estimand stays biased (see the "
            "four-number gate and the E-value / add-unobserved-common-cause sweep)."
        ),
    }


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    panel = read_panel_analysis(cfg_obj)
    out_dir = Path(cfg_obj.paths.root) / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "dag_analyst.gml").write_text(analyst_dag())
    (out_dir / "dag_true.gml").write_text(true_dag())

    naive = naive_logit_audit(panel)
    dml_a = dml_audit(panel, seed=cfg_obj.seed)
    dml_o = dml_onset(panel, seed=cfg_obj.seed)
    did = estimate_did(panel, seed=cfg_obj.seed)
    cate = cate_by_group(panel, seed=cfg_obj.seed)
    refutation = refute_audit(
        panel, seed=cfg_obj.seed, subset_fraction=cfg_obj.causal.refuter_subset_frac
    )
    payload = {
        "adjustment_set": observed_adjustment_set(),
        "identifiability": _identifiability_report(panel),
        "naive_logit_audit": asdict(naive),
        "dml_audit_observed": asdict(dml_a),
        "dml_onset": asdict(dml_o),
        "did": asdict(did),
        "cate_by_size_decile": cate.by_size_decile,
        "cate_by_sector": cate.by_sector,
        "refutation": asdict(refutation),
    }
    if cfg_obj.causal.run_discovery:
        from simworld.causal.discovery import discover

        payload["discovery"] = asdict(discover(panel, seed=cfg_obj.seed))
    (out_dir / "causal_estimates.json").write_text(json.dumps(payload, indent=2))
    log.info(
        "causal analysis done: naive=%.4f dml_audit=%.4f dml_onset=%.4f did=%.4f placebo=%.4f",
        naive.estimate,
        dml_a.estimate,
        dml_o.estimate,
        did.att,
        refutation.placebo_effect,
    )


if __name__ == "__main__":
    main()
