#!/bin/bash
#SBATCH --job-name=ART-{{METHOD}}-{{BACKEND}}   # will be overridden by submitter
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out                 # logs/ART-<name>-<jobid>.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=nmmz76@umsystem.edu

# Fail fast and print commands
set -euo pipefail

# Always start in submit directory so relative paths (venv, repo) work
cd "$SLURM_SUBMIT_DIR"

# Environment
module load python/3.12.1
source venv2/bin/activate
pip install AdaptiveResonanceLib/

# Defaults if not exported by submitter
METHOD="${METHOD:-BinaryFuzzyARTMAP}"
BACKEND="${BACKEND:-python}"
SEED="${SEED:-0}"

# Run
python AdaptiveResonanceLib/scripts/compare_factories.py \
  --method "${METHOD}" \
  --backend "${BACKEND}" \
  --seed "${SEED}"
