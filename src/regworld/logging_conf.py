"""Central logging configuration. `print()` in src/ is banned by lint; loggers only."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once; safe to call repeatedly."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
    # Chatty third parties stay at WARNING so stage logs remain readable.
    for noisy in ("matplotlib", "mlflow", "urllib3", "fsspec", "PIL", "jax", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
