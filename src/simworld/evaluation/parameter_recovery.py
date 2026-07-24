"""§11 family 7 — parameter recovery: theta* coverage, bias, z-scores (C1).

Grades the Stage-4 posterior against the sealed answer key. The full C1 claim is a
*contrast*: calibration recovers theta* when the model is well specified, and fails
legibly (a biased peer coefficient beta_peer) when supply-network capacity homophily
is switched on. A single pipeline run ships ONE dgp variant, so that contrast cannot
come from the main artifacts alone; ``scripts/recovery_grid.py`` re-calibrates under
both worlds and calls :func:`build_grid` here to grade them. When that grid is present
:func:`evaluate` reports the two-cell contrast; otherwise it grades the shipped world's
single variant and marks the other cell not-run rather than pretending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from simworld.data.store import read_oracle
from simworld.types import SimWorldConfig

GRID_CELLS = ("wellspecified", "confounded")
GRID_FILE = "recovery_grid.json"
GRID_SCHEMA = "simworld.c1.recovery_grid.v1"
COVERAGE_BAR = 12  # >= 12/16 parameters must cover theta* at 90% under wellspecified (§18)


def _grade(theta_star: dict[str, Any], idata: Any) -> dict[str, Any]:
    """90% HDI coverage of theta* for one posterior (the hidden confounder is excluded)."""
    rows = []
    covered = 0
    graded = 0
    for name, true_value in theta_star.items():
        if name not in idata.posterior.data_vars or name == "beta_capacity":
            continue
        draws = np.asarray(idata.posterior[name]).reshape(-1)
        lo, hi = np.quantile(draws, [0.05, 0.95])
        mean = float(draws.mean())
        sd = float(draws.std()) or 1e-9
        inside = bool(lo <= float(true_value) <= hi)
        graded += 1
        covered += inside
        rows.append(
            {
                "parameter": name,
                "true": round(float(true_value), 4),
                "posterior_mean": round(mean, 4),
                "bias": round(mean - float(true_value), 4),
                "z_score": round((mean - float(true_value)) / sd, 2),
                "hdi_90_covers": inside,
            }
        )
    beta_peer = next((r for r in rows if r["parameter"] == "beta_peer"), None)
    return {
        "coverage_at_90": f"{covered}/{graded}",
        "coverage_fraction": round(covered / graded, 3) if graded else None,
        "covered": covered,
        "graded": graded,
        "per_parameter": rows,
        "beta_peer_covers": bool(beta_peer and beta_peer["hdi_90_covers"]),
        "beta_peer_bias": beta_peer["bias"] if beta_peer else None,
    }


def _micro_convergence(root: Path) -> dict[str, Any]:
    """max R-hat / divergences from a cell's micro diagnostics (empty if absent)."""
    diag = root / "calibration" / "micro_diagnostics.json"
    if not diag.is_file():
        return {}
    payload = json.loads(diag.read_text())
    return {"max_r_hat": payload.get("max_r_hat"), "divergences": payload.get("divergences")}


def build_grid(cfg: SimWorldConfig, cell_roots: dict[str, str]) -> Path:
    """Grade each variant's posterior against theta* and write ``recovery_grid.json``.

    Lives here (``evaluation/``) because grading reads the sealed theta*. theta* is
    identical across variants (the behavioral truth is shared; only the world's
    structure differs), so both cells are graded against the same answer key read once
    from the main root. ``cell_roots`` maps each variant to the ``paths.root`` its
    world+data+calibrate chain wrote to.
    """
    import arviz as az

    # theta* is identical across variants; read it from a cell root (guaranteed present
    # after that cell's world was generated) so the grid does not depend on the caller's
    # root also carrying an answer key.
    ref_root = cell_roots.get("wellspecified") or next(iter(cell_roots.values()))
    ref_cfg = cfg.model_copy(deep=True)
    ref_cfg.paths.root = ref_root
    theta_star = read_oracle(ref_cfg, "theta_star")
    cells: dict[str, Any] = {}
    for variant, root in cell_roots.items():
        posterior = Path(root) / "calibration" / "posterior.nc"
        graded = _grade(theta_star, az.from_netcdf(posterior))
        graded.update(_micro_convergence(Path(root)))
        cells[variant] = graded

    ws = cells["wellspecified"]
    cf = cells["confounded"]
    ws_converged = (
        ws.get("max_r_hat") is not None
        and ws["max_r_hat"] < 1.01
        and ws.get("divergences") in (0, None)
    )
    recovers = bool(ws["covered"] >= COVERAGE_BAR and ws_converged)
    beta_peer_biased = not cf["beta_peer_covers"]
    contrast = {
        "recovers_when_wellspecified": recovers,
        "beta_peer_biased_when_confounded": beta_peer_biased,
        "clean_contrast": bool(recovers and beta_peer_biased),
        "wellspecified_coverage": ws["coverage_at_90"],
        "wellspecified_max_r_hat": ws.get("max_r_hat"),
        "confounded_beta_peer_bias": cf.get("beta_peer_bias"),
    }
    payload = {
        "schema": GRID_SCHEMA,
        "cells": cells,
        "contrast": contrast,
        "coverage_bar": f">= {COVERAGE_BAR}/16 at 90% under wellspecified",
        "note": (
            "C1 is a two-world contrast. SUPPORTED requires clean recovery under "
            "wellspecified (>= 12/16 coverage AND max R-hat < 1.01) AND a biased "
            "beta_peer under confounded. Both halves sharpen with dev-scale draws and "
            "populations; at smoke the posteriors are too wide to resolve the contrast."
        ),
    }
    out = Path(cfg.paths.root) / "calibration" / GRID_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def _load_grid(cfg: SimWorldConfig) -> dict[str, Any] | None:
    path = Path(cfg.paths.root) / "calibration" / GRID_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def evaluate(cfg: SimWorldConfig) -> dict[str, object]:
    import arviz as az

    grid = _load_grid(cfg)
    if grid is not None:
        cells = grid["cells"]
        ws = cells["wellspecified"]
        cf = cells["confounded"]
        return {
            "mode": "contrast (wellspecified vs confounded)",
            "dgp_variant": "grid",
            # C1's recovery half is judged under the WELL-SPECIFIED world...
            "coverage_at_90": ws["coverage_at_90"],
            "coverage_fraction": ws["coverage_fraction"],
            "max_r_hat": ws.get("max_r_hat"),
            "divergences": ws.get("divergences"),
            "per_parameter": ws["per_parameter"],
            # ...and its failure half under the CONFOUNDED world.
            "beta_peer_miss_under_confounded": not cf["beta_peer_covers"],
            "confounded_beta_peer_bias": cf.get("beta_peer_bias"),
            "contrast": grid["contrast"],
            "c1_grid_status": (
                "wellspecified-vs-confounded contrast produced by scripts/recovery_grid.py"
            ),
            "thresholds_dev": {
                "coverage": ">= 12/16 at 90% under wellspecified",
                "beta_peer": "must miss under confounded",
                "convergence": "max R-hat < 1.01 (needs dev-scale draws)",
            },
        }

    # Fallback: no grid — grade the shipped world's single variant (the confounded
    # world at smoke). The contrast's recovery half is unmeasured until the grid runs.
    theta_star = read_oracle(cfg, "theta_star")
    posterior_path = Path(cfg.paths.root) / "calibration" / "posterior.nc"
    if not posterior_path.is_file():
        return {"status": "posterior.nc missing; run `make calibrate`"}
    graded = _grade(theta_star, az.from_netcdf(posterior_path))
    variant = cfg.dgp.variant
    return {
        "mode": f"single variant ({variant})",
        "dgp_variant": variant,
        "coverage_at_90": graded["coverage_at_90"],
        "coverage_fraction": graded["coverage_fraction"],
        "per_parameter": graded["per_parameter"],
        "beta_peer_miss_under_confounded": (
            (not graded["beta_peer_covers"])
            if variant == "confounded"
            else "n/a (variant is not confounded)"
        ),
        "c1_grid_status": (
            f"this run grades the shipped '{variant}' world; the wellspecified-vs-"
            "confounded contrast needs `calibration.recovery_grid=true` "
            "(scripts/recovery_grid.py; on at dev, off at smoke for the < 6 min budget)"
        ),
        "thresholds_dev": {
            "coverage": ">= 12/16 at 90% under wellspecified",
            "beta_peer": "must miss under confounded",
        },
    }
