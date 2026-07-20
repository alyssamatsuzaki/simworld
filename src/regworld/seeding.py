"""One seeding entry point (§13). Every stochastic component takes the returned Generator.

JAX is deliberately not imported here: calibration runs JAX in a subprocess (§5) and seeds
its own PRNGKey there. If JAX happens to be loaded already, its key helper is available via
`jax_key`.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy (legacy + Generator), and torch (CPU/CUDA if present).

    Returns a `np.random.Generator` to be passed explicitly; no bare `np.random.*`
    calls are permitted anywhere in src/ (§13).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberately seeds legacy global state for 3rd-party libs
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU-first (§5)
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is core, but stay import-safe
        pass
    return np.random.default_rng(seed)


def jax_key(seed: int) -> object:
    """PRNG key for JAX stages; import stays local so the main process never loads JAX."""
    import jax

    return jax.random.PRNGKey(seed)


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split a Generator into n independent child streams."""
    return [np.random.default_rng(s) for s in rng.spawn(n)]
