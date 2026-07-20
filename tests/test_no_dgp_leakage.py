"""THE FIREWALL (§1). The answer key lives in `regworld.dgp` and `artifacts/oracle/`.

Only three kinds of code may touch it:
  - the world builders that WRITE it (`dgp/` itself, `data/generate.py`,
    `causal/ground_truth.py` — run at generation time, before anything is estimated),
  - `data/store.py`, which implements the guarded `read_oracle()` accessor,
  - `evaluation/`, which grades everything against it.

Any other import of `regworld.dgp` or reference to the oracle tree invalidates the
entire evaluation section. This test greps the source tree; `data/store.py::read_oracle`
adds a stack-frame check at runtime. Neither is optional.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "regworld"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

DGP_IMPORT = re.compile(
    r"^\s*(?:from\s+regworld\.dgp|import\s+regworld\.dgp"
    r"|from\s+regworld\s+import\s+.*\bdgp\b|from\s+\.\s*import\s+.*\bdgp\b)",
    re.MULTILINE,
)
ORACLE_REF = re.compile(r"oracle", re.IGNORECASE)

DGP_ALLOWED = {
    "dgp",  # the package itself
    "data/generate.py",  # writes observed/ + oracle/ at world-build time
    "causal/ground_truth.py",  # runs do() in the DGP at world-build time
    "evaluation",  # grades against the answer key, by design
}
ORACLE_ALLOWED = DGP_ALLOWED | {"data/store.py"}  # store.py implements the guarded accessor


def _allowed(rel: str, allowlist: set[str]) -> bool:
    return any(rel == a or rel.startswith(a + "/") for a in allowlist)


def test_no_dgp_import_outside_allowlist() -> None:
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if _allowed(rel, DGP_ALLOWED):
            continue
        if DGP_IMPORT.search(p.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == [], f"regworld.dgp imported outside the firewall: {offenders}"


def test_no_oracle_reference_outside_allowlist() -> None:
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if _allowed(rel, ORACLE_ALLOWED):
            continue
        if ORACLE_REF.search(p.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == [], f"oracle referenced outside the firewall: {offenders}"


def test_scripts_only_world_builders_touch_dgp() -> None:
    allowed = {"generate_world.py"}
    offenders = []
    for p in SCRIPTS.glob("*.py"):
        if p.name in allowed:
            continue
        if DGP_IMPORT.search(p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    assert offenders == [], f"scripts importing regworld.dgp: {offenders}"
