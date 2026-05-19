#!/bin/bash
#SBATCH --job-name=ART-{{METHOD}}-{{BACKEND}}-{{DEVICE}}   # will be overridden
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j.out                 # logs/ART-<name>-<jobid>.out
#SBATCH --constraint=intel&skylake&40CPU
#SBATCH --gres=gpu:V100-SXM2-32GB:1

# Fail fast and print commands
set -euo pipefail

# Always start in submit directory so relative paths (venv, repo) work
cd "$SLURM_SUBMIT_DIR"

# Environment
module load python/3.12.1
source venv2/bin/activate
mkdir -p "torchdata/"
export TORCH_HOME="torchdata/"

# Defaults if not exported by submitter
METHOD="${METHOD:-BinaryFuzzyARTMAP}"
BACKEND="${BACKEND:-torch}"
DEVICE="${DEVICE:-gpu}"
SEED="${SEED:-0}"

# Run
python AdaptiveResonanceLib/scripts/compare_factories.py \
  --method "${METHOD}" \
  --backend "${BACKEND}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
