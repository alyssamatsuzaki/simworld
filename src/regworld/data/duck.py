"""DuckDB views over the observed Parquet directory (§8): free SQL, no database
to stand up. Used by notebooks and the dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from regworld.data.store import observed_dir
from regworld.types import RegWorldConfig

log = logging.getLogger(__name__)


def build_views(cfg: RegWorldConfig) -> Path:
    d = observed_dir(cfg)
    db_path = d / "views.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        for pq in sorted(d.glob("*.parquet")):
            con.execute(
                f"CREATE OR REPLACE VIEW {pq.stem} AS SELECT * FROM read_parquet('{pq.as_posix()}')"
            )
        gdir = d / "graphs"
        for pq in sorted(gdir.glob("*.parquet")) if gdir.exists() else []:
            con.execute(
                f"CREATE OR REPLACE VIEW {pq.stem} AS SELECT * FROM read_parquet('{pq.as_posix()}')"
            )
        n = con.execute("SELECT count(*) FROM duckdb_views() WHERE NOT internal").fetchone()
        log.info("duckdb views built at %s (%s views)", db_path, n[0] if n else "?")
    finally:
        con.close()
    return db_path
