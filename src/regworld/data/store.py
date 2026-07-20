"""Parquet/Zarr/JSON store with the oracle read guard (§1).

`read_oracle` raises unless the caller is `regworld.evaluation`, a test, or a
world builder — a runtime stack-frame check that backs up the grep test in
`tests/test_no_dgp_leakage.py`. Neither is optional.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl

from regworld.data.schema import ALL_OBSERVED, TableSpec, validate_table
from regworld.types import RegWorldConfig

_ORACLE_ALLOWED_FRAGMENTS = (
    "regworld/evaluation/",
    "regworld/data/generate.py",
    "regworld/causal/ground_truth.py",
    "/tests/",
    "site-packages/pytest",
)


def observed_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.data) / "observed"


def oracle_dir(cfg: RegWorldConfig) -> Path:
    return Path(cfg.paths.root) / "oracle"


def write_observed(
    cfg: RegWorldConfig, name: str, df: pl.DataFrame, spec: TableSpec | None = None
) -> Path:
    spec = spec or ALL_OBSERVED.get(name)
    if spec is not None:
        validate_table(df, spec)
    d = observed_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.parquet"
    df.write_parquet(path, compression="snappy")
    return path


def read_observed(cfg: RegWorldConfig, name: str, validate: bool = True) -> pl.DataFrame:
    path = observed_dir(cfg) / f"{name}.parquet"
    df = pl.read_parquet(path)
    spec = ALL_OBSERVED.get(name)
    if validate and spec is not None:
        validate_table(df, spec)
    return df


def _caller_is_allowed() -> bool:
    """Check the IMMEDIATE caller of read_oracle only: a disallowed module cannot
    launder an oracle read by being invoked from an allowed one."""
    stack = inspect.stack()
    if len(stack) < 3:  # pragma: no cover - interactive use
        return True
    fname = stack[2].filename.replace("\\", "/")
    return any(fragment in fname for fragment in _ORACLE_ALLOWED_FRAGMENTS)


def read_oracle(cfg: RegWorldConfig, name: str) -> Any:
    """Read from the answer key. Raises RuntimeError for unauthorized callers (§1)."""
    if not _caller_is_allowed():
        offender = inspect.stack()[1].filename
        raise RuntimeError(
            f"oracle read from {offender!r}: only regworld.evaluation, tests, and "
            "world builders may read artifacts/oracle (PLAN.md §1 firewall)"
        )
    d = oracle_dir(cfg)
    if (d / f"{name}.parquet").exists():
        return pl.read_parquet(d / f"{name}.parquet")
    if (d / f"{name}.json").exists():
        return json.loads((d / f"{name}.json").read_text())
    if (d / f"{name}.zarr").exists():
        import xarray as xr

        return xr.open_zarr(d / f"{name}.zarr")
    raise FileNotFoundError(f"no oracle artifact named {name!r} under {d}")


def write_oracle_parquet(cfg: RegWorldConfig, name: str, df: pl.DataFrame) -> Path:
    d = oracle_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.parquet"
    df.write_parquet(path, compression="snappy")
    return path


def write_oracle_json(cfg: RegWorldConfig, name: str, payload: dict[str, Any]) -> Path:
    d = oracle_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
