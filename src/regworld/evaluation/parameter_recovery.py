"""§11 family 7 — parameter recovery: theta* coverage, bias, z-scores (C1).

Grades the Stage-4 posterior against the sealed answer key. The full C1 grid
(wellspecified vs confounded reruns of the entire calibration) is a dev-profile
run; at smoke this reports the shipped world's variant and marks the other grid
cell not-run rather than pretending.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from regworld.data.store import read_oracle
from regworld.types import RegWorldConfig


def evaluate(cfg: RegWorldConfig) -> dict[str, object]:
    import arviz as az

    theta_star = read_oracle(cfg, "theta_star")
    posterior_path = Path(cfg.paths.root) / "calibration" / "posterior.nc"
    if not posterior_path.is_file():
        return {"status": "posterior.nc missing; run `make calibrate`"}
    idata = az.from_netcdf(posterior_path)
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
    variant = cfg.dgp.variant
    beta_peer_row = next((r for r in rows if r["parameter"] == "beta_peer"), None)
    return {
        "dgp_variant": variant,
        "coverage_at_90": f"{covered}/{graded}",
        "coverage_fraction": round(covered / graded, 3) if graded else None,
        "per_parameter": rows,
        "beta_peer_miss_under_confounded": (
            bool(beta_peer_row and not beta_peer_row["hdi_90_covers"])
            if variant == "confounded"
            else "n/a (variant is not confounded)"
        ),
        "c1_grid_status": (
            f"this run grades the shipped '{variant}' world; the "
            "wellspecified-vs-confounded contrast is a dev-profile rerun of Stage 4 "
            "(not run at smoke)"
        ),
        "thresholds_dev": {
            "coverage": ">= 12/16 at 90% under wellspecified",
            "beta_peer": "must miss under confounded",
        },
    }
