"""THE FIREWALL (§1). The answer key lives in `regworld.dgp` and `artifacts/oracle/`.

Only three kinds of code may touch it:
  - the world builders that WRITE it (`dgp/` itself, `data/generate.py`,
    `causal/ground_truth.py` — run at generation time, before anything is estimated),
  - `data/store.py`, which implements the guarded `read_oracle()` accessor,
  - `evaluation/`, which grades everything against it.

Any other import of `regworld.dgp` or reference to the oracle tree invalidates the
entire evaluation section. This test greps the source tree AND the scripts/ entry
points; `data/store.py::read_oracle` adds a stack-frame check at runtime. Neither
is optional.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "regworld"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# Static import forms: absolute (`import regworld.dgp`, `from regworld.dgp import`,
# `from regworld import dgp`) and relative (`from .dgp import`, `from ..dgp import`,
# `from . import dgp`, `from .. import dgp`) — a relative import inside
# src/regworld/*/ reaches the same package without ever spelling "regworld.dgp".
DGP_IMPORT = re.compile(
    r"^\s*(?:from\s+regworld\.dgp"
    r"|import\s+regworld\.dgp"
    r"|from\s+regworld\s+import\s+.*\bdgp\b"
    r"|from\s+\.{1,2}\s*dgp\b"
    r"|from\s+\.{1,2}\s*import\s+.*\bdgp\b)",
    re.MULTILINE,
)
# Dynamic/string-built imports: `importlib.import_module("regworld.dgp...")`,
# `__import__("regworld.dgp")`, and obvious concatenations such as
# `import_module("regworld" + ".dgp")`. `[^)]*` matches newlines (negated class),
# so multi-line call sites are caught too. Exotic laundering (f-strings, getattr
# chains) is out of regex reach; `read_oracle`'s stack check is the runtime
# backstop for those.
DGP_DYNAMIC = re.compile(
    r"(?:\bimport_module|\b__import__)\s*\([^)]*(?:regworld\.dgp|['\"]\s*\.?dgp\b)"
)
ORACLE_REF = re.compile(r"oracle", re.IGNORECASE)

DGP_ALLOWED = {
    "dgp",  # the package itself
    "data/generate.py",  # writes observed/ + oracle/ at world-build time
    "causal/ground_truth.py",  # runs do() in the DGP at world-build time
    "evaluation",  # grades against the answer key, by design
}
ORACLE_ALLOWED = DGP_ALLOWED | {"data/store.py"}  # store.py implements the guarded accessor

# scripts/generate_world.py is Stage 1a's entry point: it calls the sanctioned
# world builder `regworld.data.generate.generate_ground_truth`, which writes BOTH
# trees (observed/ and oracle/) at generation time, before anything is estimated.
# It is therefore the only script allowed to import the DGP or name the oracle.
SCRIPT_DGP_ALLOWED = {"generate_world.py"}
SCRIPT_ORACLE_ALLOWED = {"generate_world.py"}


def _allowed(rel: str, allowlist: set[str]) -> bool:
    return any(rel == a or rel.startswith(a + "/") for a in allowlist)


def _imports_dgp(source: str) -> bool:
    return bool(DGP_IMPORT.search(source) or DGP_DYNAMIC.search(source))


def test_no_dgp_import_outside_allowlist() -> None:
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if _allowed(rel, DGP_ALLOWED):
            continue
        if _imports_dgp(p.read_text(encoding="utf-8")):
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
    offenders = []
    for p in SCRIPTS.rglob("*.py"):
        if p.name in SCRIPT_DGP_ALLOWED:
            continue
        if _imports_dgp(p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    assert offenders == [], f"scripts importing regworld.dgp: {offenders}"


def test_scripts_do_not_reference_oracle() -> None:
    """A script reading `artifacts/oracle/...` directly would bypass both the src
    grep and `read_oracle`'s stack check — so scripts get the same oracle sweep."""
    offenders = []
    for p in SCRIPTS.rglob("*.py"):
        if p.name in SCRIPT_ORACLE_ALLOWED:
            continue
        if ORACLE_REF.search(p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    assert offenders == [], f"scripts referencing the oracle tree: {offenders}"


def test_firewall_regexes_catch_known_evasions() -> None:
    """The tripwire itself is tested: every known evasion form must trip it."""
    evasions = [
        "from regworld.dgp import world",
        "import regworld.dgp",
        "import regworld.dgp.world as w",
        "from regworld import dgp",
        "from regworld import data, dgp",
        "from .dgp import world",
        "from ..dgp import world",
        "from . import dgp",
        "from .. import dgp",
        'importlib.import_module("regworld.dgp")',
        "import_module('regworld.dgp.world')",
        '__import__("regworld.dgp")',
        'import_module("regworld" + ".dgp")',
        'import_module(\n    "regworld.dgp"\n)',
    ]
    for snippet in evasions:
        assert _imports_dgp(snippet), f"firewall regex missed: {snippet!r}"
    innocents = [
        "from regworld.data import store",
        "from .dgp_free_module import helper",  # 'dgp' prefix of a longer name
        "__import__(mod).__version__",  # stages.py version probe, no literal
        "# the DGP binds theta-star in regworld/dgp (docstring mention, no import)",
    ]
    for snippet in innocents:
        assert not DGP_IMPORT.search(snippet), f"false positive (import): {snippet!r}"
        assert not DGP_DYNAMIC.search(snippet), f"false positive (dynamic): {snippet!r}"


def test_estimated_theta_defaults_do_not_reveal_answer_key() -> None:
    from regworld.rules import Theta

    assert Theta().beta_peer != 1.4
    assert Theta().beta_capacity == 0.0
