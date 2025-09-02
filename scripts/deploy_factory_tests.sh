#!/usr/bin/env bash
set -euo pipefail

# Where the template lives (assume same dir as this script)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TPL="${SCRIPT_DIR}/run_template.slurm"

# Slurm options common to all submissions
PARTITION="general"
CONSTRAINT=""                 # e.g., "intel&skylake&40CPU" or "amd&64CPU"; leave empty to skip
SEED=0

METHODS=( BinaryFuzzyARTMAP FuzzyARTMAP HypersphereARTMAP GaussianARTMAP )
BACKENDS=( python torch "c++" )

mkdir -p logs

for method in "${METHODS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    # Sanitize job name (no '+')
    bn="${backend/c++/cpp}"
    jobname="ART-${method}-${bn}"

    echo "[SUBMIT] ${method} / ${backend}"

    # Build sbatch args (override job name & logs; export vars the template uses)
    args=(
      --job-name="${jobname}"
      --partition="${PARTITION}"
      --output="logs/%x-%j.out"
      --error="logs/%x-%j.err"
      --export=ALL,METHOD="${method}",BACKEND="${backend}",SEED="${SEED}"
    )
    [[ -n "${CONSTRAINT}" ]] && args+=( --constraint="${CONSTRAINT}" )

    sbatch "${args[@]}" "${TPL}"
  done
done
