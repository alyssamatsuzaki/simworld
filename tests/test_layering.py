"""Structural rule (§4): src/ never imports from notebooks/."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "regworld"


def test_src_never_imports_notebooks() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+notebooks\b", re.MULTILINE)
    offenders = [str(p) for p in SRC.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], f"src/ imports notebooks/: {offenders}"
