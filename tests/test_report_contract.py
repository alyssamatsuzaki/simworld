"""Test the report contract: build_findings produces FINDINGS.md with required sections."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from regworld.evaluation.report import build_findings
from regworld.types import PathsCfg, RegWorldConfig, StagesCfg


def test_build_findings_produces_findings_md():
    """build_findings writes reports/FINDINGS.md and returns its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg = RegWorldConfig(
            seed=0,
            profile_name="smoke",
            paths=PathsCfg(
                root=str(tmpdir_path / "artifacts"),
                reports=str(tmpdir_path / "reports"),
            ),
            stages=StagesCfg(),
        )

        # Create minimal artifact structure
        (tmpdir_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "reports" / "eval").mkdir(parents=True, exist_ok=True)

        # Write stub artifacts (empty dicts)
        (tmpdir_path / "artifacts" / "causal").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal" / "four_numbers.json").write_text("{}")
        (tmpdir_path / "artifacts" / "sensitivity").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "sensitivity" / "sensitivity_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "ensemble").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "ensemble" / "ensemble_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "calibration").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "calibration" / "micro_diagnostics.json").write_text("{}")
        (tmpdir_path / "reports" / "eval" / "metrics.json").write_text("{}")
        (tmpdir_path / "reports" / "run_manifest.json").write_text("{}")

        result_path = build_findings(cfg)

        assert result_path.exists()
        assert result_path.name == "FINDINGS.md"
        assert result_path.read_text()  # Non-empty
        assert len(result_path.read_text()) > 0


def test_build_findings_contains_required_heading_where_this_model_fails():
    """The generated report CONTAINS the required 'Where this model fails' heading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg = RegWorldConfig(
            seed=0,
            profile_name="smoke",
            paths=PathsCfg(
                root=str(tmpdir_path / "artifacts"),
                reports=str(tmpdir_path / "reports"),
            ),
            stages=StagesCfg(),
        )

        (tmpdir_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "reports" / "eval").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal" / "four_numbers.json").write_text("{}")
        (tmpdir_path / "artifacts" / "sensitivity").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "sensitivity" / "sensitivity_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "ensemble").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "ensemble" / "ensemble_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "calibration").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "calibration" / "micro_diagnostics.json").write_text("{}")
        (tmpdir_path / "reports" / "eval" / "metrics.json").write_text("{}")
        (tmpdir_path / "reports" / "run_manifest.json").write_text("{}")

        result_path = build_findings(cfg)
        content = result_path.read_text()

        # The heading MUST be present.
        assert "## Where This Model Fails" in content, (
            "The required heading '## Where This Model Fails' is missing from FINDINGS.md. "
            "This section is client-critical and must always be present."
        )


def test_disclaimer_precedes_claims():
    """The disclaimer appears before any C1-C6 verdict in the report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg = RegWorldConfig(
            seed=0,
            profile_name="smoke",
            paths=PathsCfg(
                root=str(tmpdir_path / "artifacts"),
                reports=str(tmpdir_path / "reports"),
            ),
            stages=StagesCfg(),
        )

        (tmpdir_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "reports" / "eval").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal" / "four_numbers.json").write_text("{}")
        (tmpdir_path / "artifacts" / "sensitivity").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "sensitivity" / "sensitivity_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "ensemble").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "ensemble" / "ensemble_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "calibration").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "calibration" / "micro_diagnostics.json").write_text("{}")
        (tmpdir_path / "reports" / "eval" / "metrics.json").write_text("{}")
        (tmpdir_path / "reports" / "run_manifest.json").write_text("{}")

        result_path = build_findings(cfg)
        content = result_path.read_text()

        # Find byte offsets
        disclaimer_pos = content.find("## Disclaimer")
        c1_pos = content.find("### C1")

        assert disclaimer_pos != -1, "Disclaimer section not found"
        assert c1_pos != -1, "C1 claim section not found"
        assert disclaimer_pos < c1_pos, "Disclaimer must appear before claims"


def test_four_number_table_labels_present():
    """The generated report contains the four-number table labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg = RegWorldConfig(
            seed=0,
            profile_name="smoke",
            paths=PathsCfg(
                root=str(tmpdir_path / "artifacts"),
                reports=str(tmpdir_path / "reports"),
            ),
            stages=StagesCfg(),
        )

        (tmpdir_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "reports" / "eval").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "causal" / "four_numbers.json").write_text(
            json.dumps(
                {
                    "tau_true": 0.415,
                    "tau_abm": 0.347,
                    "tau_qe": 0.061,
                    "tau_qe_ci": [-0.113, 0.262],
                    "tau_obs": 0.125,
                    "tau_obs_ci": [0.031, 0.218],
                }
            )
        )
        (tmpdir_path / "artifacts" / "sensitivity").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "sensitivity" / "sensitivity_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "ensemble").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "ensemble" / "ensemble_summary.json").write_text("{}")
        (tmpdir_path / "artifacts" / "calibration").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "artifacts" / "calibration" / "micro_diagnostics.json").write_text("{}")
        (tmpdir_path / "reports" / "eval" / "metrics.json").write_text("{}")
        (tmpdir_path / "reports" / "run_manifest.json").write_text("{}")

        result_path = build_findings(cfg)
        content = result_path.read_text()

        # All four labels should be present
        assert "τ_true" in content, "τ_true label missing"
        assert "τ_abm" in content, "τ_abm label missing"
        assert "τ_qe" in content, "τ_qe label missing"
        assert "τ_obs" in content, "τ_obs label missing"


def test_graceful_degradation_missing_artifacts():
    """If artifacts are missing, the report is written with explicit missing-artifact notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg = RegWorldConfig(
            seed=0,
            profile_name="smoke",
            paths=PathsCfg(
                root=str(tmpdir_path / "artifacts"),
                reports=str(tmpdir_path / "reports"),
            ),
            stages=StagesCfg(),
        )

        # Create minimal structure but omit the key artifacts
        (tmpdir_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmpdir_path / "reports" / "eval").mkdir(parents=True, exist_ok=True)

        result_path = build_findings(cfg)

        assert result_path.exists()
        content = result_path.read_text()
        # The report should still be written, possibly with artifact-not-found notes
        assert len(content) > 0
        # It should indicate artifacts are missing
        assert "missing" in content.lower() or "not found" in content.lower()


@pytest.mark.skip(reason="Requires committed artifacts from a full run")
def test_build_findings_on_committed_artifacts():
    """Integration test: build_findings on committed artifacts (smoke profile)."""
    cfg = RegWorldConfig(
        seed=0,
        profile_name="smoke",
        paths=PathsCfg(root="artifacts", reports="reports"),
        stages=StagesCfg(),
    )

    result_path = build_findings(cfg)

    assert result_path.exists()
    content = result_path.read_text()

    # All five required sections
    assert "## Disclaimer" in content
    assert "## The Four-Number Causal Table" in content
    assert "## The Six Claims" in content
    assert "## Where This Model Fails" in content
    assert "## Run Manifest" in content

    # Claims
    for claim_key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        assert f"### {claim_key}" in content, f"Claim {claim_key} missing"
