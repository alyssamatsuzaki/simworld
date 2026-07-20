"""Stage 5a-5e: identify, estimate four ways, refute, discover — all on observed data."""

import json
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

from regworld.causal.did import estimate_did
from regworld.causal.estimate import cate_by_group, dml_audit, dml_onset, naive_logit_audit
from regworld.causal.graph import analyst_dag, observed_adjustment_set, true_dag
from regworld.causal.refute import refute_audit
from regworld.data.ingest import read_panel_analysis
from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = get_logger(__name__)


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
        "naive_logit_audit": asdict(naive),
        "dml_audit_observed": asdict(dml_a),
        "dml_onset": asdict(dml_o),
        "did": asdict(did),
        "cate_by_size_decile": cate.by_size_decile,
        "cate_by_sector": cate.by_sector,
        "refutation": asdict(refutation),
    }
    if cfg_obj.causal.run_discovery:
        from regworld.causal.discovery import discover

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
