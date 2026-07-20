"""§11 family 8 — the four-number causal table, restated from the Stage-5f gate."""

from __future__ import annotations

import json
from pathlib import Path

from regworld.types import RegWorldConfig


def evaluate(cfg: RegWorldConfig) -> dict[str, object]:
    path = Path(cfg.paths.root) / "causal" / "four_numbers.json"
    if not path.is_file():
        return {"status": "four_numbers.json missing; run `make causal`"}
    numbers = json.loads(path.read_text())
    return {
        "tau_true": numbers["tau_true"],
        "tau_did_truth": numbers["tau_did_truth"],
        "tau_abm": numbers["tau_abm"],
        "tau_qe": numbers["tau_qe"],
        "tau_qe_ci": numbers["tau_qe_ci"],
        "tau_obs": numbers["tau_obs"],
        "tau_obs_ci": numbers["tau_obs_ci"],
        "gate_flagged": numbers["flagged"],
        "verdicts": {
            "sign_ok": numbers["sign_ok"],
            "magnitude_ok": numbers["magnitude_ok"],
            "did_agreement_ok": numbers["did_agreement_ok"],
        },
        "thresholds_dev": {
            "did": "de-attenuated CI covers its sealed estimand (tau_did_truth)",
            "dml": "tight CI around the wrong number (does not cover tau_true)",
        },
    }
