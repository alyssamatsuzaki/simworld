#!/bin/bash
#SBATCH --job-name=regworld
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
module purge 2>/dev/null || true
export MPLBACKEND=Agg JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONHASHSEED=0
export REGWORLD_ARTIFACT_ROOT="${REGWORLD_ARTIFACT_ROOT:-$SCRATCH/regworld/artifacts}"
uv run python scripts/run_pipeline.py profile="${PROFILE:-full}" compute=slurm "$@"
