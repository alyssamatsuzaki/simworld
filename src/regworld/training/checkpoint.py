"""Self-contained emulator checkpoints: weights + everything needed to rebuild.

A checkpoint carries the constructor arguments, static node features, graph
template, initial-frame arrays, and the aggregate layout, so ``EmulatorEnv`` and
the evaluation suite can load a model without re-reading the training corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from regworld.models.gnn import GraphTemplate
from regworld.models.world_model import WorldModel


def save_checkpoint(
    path: Path,
    model: WorldModel,
    *,
    constructor: dict[str, Any],
    static_features: dict[str, torch.Tensor],
    template: GraphTemplate,
    initial: dict[str, torch.Tensor],
    aggregate_names: list[str],
    extras: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "constructor": constructor,
            "static_features": {k: v.cpu() for k, v in static_features.items()},
            "template_counts": template.node_counts,
            "template_edges": {"__".join(k): v.cpu() for k, v in template.edge_index.items()},
            "initial": {k: v.cpu() for k, v in initial.items()},
            "aggregate_names": aggregate_names,
            "extras": extras,
        },
        path,
    )
    return path


def load_checkpoint(path: Path) -> tuple[WorldModel, dict[str, Any]]:
    """Rebuild the model and return it in eval mode with its metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"emulator checkpoint not found: {path}; run `make emulator`")
    payload = torch.load(path, weights_only=False, map_location="cpu")
    edges = {
        cast(tuple[str, str, str], tuple(k.split("__"))): v
        for k, v in payload["template_edges"].items()
    }
    template = GraphTemplate(payload["template_counts"], edges)
    model = WorldModel(
        static_features=payload["static_features"],
        template=template,
        **payload["constructor"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {
        "initial": payload["initial"],
        "aggregate_names": payload["aggregate_names"],
        "extras": payload["extras"],
    }
    return model, meta


def checkpoint_path(root: str | Path, arch: str) -> Path:
    return Path(root) / "emulator" / arch / "model.pt"
