"""Shared access to Stage 4's combined posterior artifact (``posterior.nc``).

Both the causal gate (5f) and the environment oracle factory bind the same
posterior-mean theta, so the artifact path and the reduction live here once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simworld.rules import Theta
from simworld.types import SimWorldConfig


def posterior_path(cfg: SimWorldConfig) -> Path:
    """Location of the combined Stage-4 posterior written by ``make calibrate``."""
    return Path(cfg.paths.root) / "calibration" / "posterior.nc"


def posterior_mean_theta(path: Path) -> Theta:
    """Posterior-mean :class:`Theta` from a ``posterior.nc`` InferenceData file.

    ``beta_capacity`` stays at its 0.0 default: the latent capacity confounder
    is never fitted, so no estimated world may bind it.
    """
    import arviz as az

    idata = az.from_netcdf(path)
    means: dict[str, float] = {}
    for name in idata.posterior.data_vars:
        if name in Theta.__dataclass_fields__:
            means[str(name)] = float(np.asarray(idata.posterior[name]).mean())
    return Theta(**means)
