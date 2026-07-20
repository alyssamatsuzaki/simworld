"""Stage 2: observed edge lists → validated PyG HeteroData artifact."""

import json
from pathlib import Path

import hydra
import polars as pl
import torch
from omegaconf import DictConfig

from regworld.data.store import observed_dir
from regworld.graphs.to_pyg import (
    hetero_from_edges,
    node_feature_contract,
    static_node_features,
)
from regworld.logging_conf import get_logger, setup_logging
from regworld.seeding import seed_everything
from regworld.types import validate_config

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    gdir = observed_dir(cfg_obj) / "graphs"
    edges = {p.stem: pl.read_parquet(p) for p in sorted(gdir.glob("*.parquet"))}
    registry = pl.read_parquet(observed_dir(cfg_obj) / "firm_registry.parquet")
    survey = pl.read_parquet(observed_dir(cfg_obj) / "consumer_survey.parquet")
    feats = static_node_features(cfg_obj, registry, survey)
    data = hetero_from_edges(cfg_obj, edges, feats)
    contract = node_feature_contract(cfg_obj)
    out_dir = Path(cfg_obj.paths.graphs)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "hetero_observed.pt")
    summary = {
        "node_types": {k: int(data[k].x.shape[0]) for k in data.node_types},
        "node_features": {
            node_type: {
                "static_dim": spec.static_dim,
                "static_names": list(spec.static),
                "phase5_dynamic_dim": spec.dynamic_dim,
                "phase5_dynamic_names": list(spec.dynamic),
            }
            for node_type, spec in contract.items()
        },
        "edge_types": {"__".join(et): int(data[et].edge_index.shape[1]) for et in data.edge_types},
    }
    (out_dir / "hetero_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("HeteroData saved: %s", summary)


if __name__ == "__main__":
    main()
