"""Stage 4: isolated micro inference, macro SMC-ABC, and durable diagnostics."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from regworld.logging_conf import setup_logging
from regworld.rules import Theta
from regworld.seeding import seed_everything
from regworld.types import RegWorldConfig, validate_config

log = logging.getLogger(__name__)


def _split_panel(data: Any) -> tuple[Any, Any]:
    quarters = np.unique(data.quarter)
    n_holdout = min(2, max(1, quarters.size // 4))
    split = quarters[-n_holdout]
    train = data.subset(data.quarter < split)
    heldout = data.subset(data.quarter >= split)
    if train.n < 20 or heldout.n == 0:
        raise ValueError("observed panel is too short for a held-out predictive check")
    return train, heldout


def _micro_worker(config_path: Path) -> None:
    """JAX/PyMC worker; invoked as a process separate from Torch simulation."""
    import arviz as az

    from regworld.calibration.diagnostics import run_micro_diagnostics
    from regworld.calibration.micro_numpyro import fit_micro_numpyro, load_micro_data
    from regworld.calibration.micro_pymc import compare_marginals, fit_micro_pymc

    cfg = RegWorldConfig.model_validate_json(config_path.read_text())
    output_dir = Path(cfg.paths.root) / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_micro_data(cfg)
    train, heldout = _split_panel(data)
    idata, mcmc = fit_micro_numpyro(
        train,
        seed=cfg.seed + 4_001,
        warmup=cfg.calibration.nuts.warmup,
        draws=cfg.calibration.nuts.draws,
        chains=cfg.calibration.nuts.chains,
    )
    micro_path = output_dir / "micro_posterior.nc"
    idata.to_netcdf(micro_path)
    diagnostics, diagnostic_paths = run_micro_diagnostics(
        idata,
        mcmc,
        train,
        heldout,
        seed=cfg.seed + 4_002,
        output_dir=output_dir,
    )
    outputs = [micro_path, *diagnostic_paths]
    crosscheck_payload: dict[str, Any] = {"enabled": cfg.calibration.crosscheck}
    if cfg.calibration.crosscheck:
        pymc_idata = fit_micro_pymc(
            train,
            seed=cfg.seed + 4_003,
            warmup=cfg.calibration.nuts.warmup,
            draws=cfg.calibration.nuts.draws,
            chains=cfg.calibration.nuts.chains,
        )
        pymc_path = output_dir / "micro_pymc_crosscheck.nc"
        pymc_idata.to_netcdf(pymc_path)
        crosscheck_payload.update(compare_marginals(idata, pymc_idata))
        outputs.append(pymc_path)
    crosscheck_path = output_dir / "crosscheck.json"
    crosscheck_path.write_text(json.dumps(crosscheck_payload, indent=2))
    outputs.append(crosscheck_path)
    manifest = {
        "outputs": [str(path) for path in outputs],
        "train_rows": train.n,
        "heldout_rows": heldout.n,
        "diagnostics": diagnostics,
        "crosscheck": crosscheck_payload,
    }
    (output_dir / "micro_worker_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Verify the durable artifact is independently readable before process exit.
    az.from_netcdf(micro_path)


def _micro_means(micro: Any) -> dict[str, float]:
    from regworld.calibration.micro_numpyro import MICRO_PARAMETER_NAMES

    return {name: float(np.asarray(micro.posterior[name]).mean()) for name in MICRO_PARAMETER_NAMES}


def _run(cfg: RegWorldConfig) -> list[Path]:
    import arviz as az

    from regworld.calibration.diagnostics import combine_posteriors
    from regworld.calibration.macro_smc import MACRO_PARAMETER_NAMES, fit_macro_smc
    from regworld.calibration.summaries import summary_statistics

    output_dir = Path(cfg.paths.root) / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    config_path.write_text(cfg.model_dump_json(indent=2))
    worker_env = dict(os.environ)
    worker_env["JAX_PLATFORMS"] = "cpu" if cfg.calibration.device == "cpu" else "gpu"
    worker_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--micro-worker", str(config_path)],
        check=True,
        env=worker_env,
    )

    micro_path = output_dir / "micro_posterior.nc"
    micro = az.from_netcdf(micro_path)
    theta = replace(Theta(), **_micro_means(micro))
    aggregate = (
        pl.read_parquet(Path(cfg.paths.data) / "observed" / "aggregate_series.parquet")
        .sort("quarter")
        .head(cfg.observed_quarters)
    )
    observed_summary = summary_statistics(aggregate)
    macro = fit_macro_smc(
        cfg,
        observed_summary,
        base_theta=theta,
        aggregate=aggregate,
        output_dir=output_dir,
    )
    combined = combine_posteriors(micro, macro, seed=cfg.seed + 4_500)
    posterior_path = output_dir / "posterior.nc"
    combined.to_netcdf(posterior_path)
    summary = az.summary(
        combined,
        var_names=[*_micro_means(micro), *MACRO_PARAMETER_NAMES],
        hdi_prob=0.90,
        kind="stats",
    )
    combined_summary_path = output_dir / "posterior_summary.csv"
    summary.to_csv(combined_summary_path)

    worker_manifest = json.loads((output_dir / "micro_worker_manifest.json").read_text())
    outputs = [
        config_path,
        posterior_path,
        combined_summary_path,
        output_dir / "macro_posterior.nc",
        output_dir / "macro_tensor_design.npz",
        output_dir / "macro_diagnostics.json",
        output_dir / "micro_worker_manifest.json",
        *[Path(path) for path in worker_manifest["outputs"]],
    ]
    outputs = list(dict.fromkeys(outputs))
    manifest = {
        "status": "DONE",
        "fitted_parameter_count": 17,
        "coverage_compatibility_threshold": 12,
        "parameter_count_note": (
            "The parameter tables specify 11 fitted micro quantities including q0/q1 "
            "and 6 macro quantities; all 17 are retained."
        ),
        "micro_backend": "numpyro_nuts",
        "micro_process_isolated": True,
        "macro_backend": "smc_abc_surrogate_over_tensorized_design",
        "outputs": [str(path) for path in outputs],
    }
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    outputs.append(manifest_path)
    return outputs


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)
    seed_everything(cfg_obj.seed)
    setup_logging()
    outputs = _run(cfg_obj)
    log.info("calibration complete: %d durable outputs", len(outputs))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--micro-worker":
        setup_logging()
        _micro_worker(Path(sys.argv[2]))
    else:
        main()
