#!/usr/bin/env bash
set -euo pipefail

# -------- child job config ----------
PARTITION="general"
CONSTRAINT="intel&skylake&40CPU"
# choose one: amd&64CPU | amd&128CPU | intel&skylake&40CPU
TIME_LIMIT="0-12:00:00"
MEM_PER_JOB="16G"
CPUS_PER_TASK=8
SEED=0
OUT_DIR="results"
LOAD_ENV_CMD="module load python/3.12.1"
# e.g., 'module load python/3.11' or 'source /path/to/venv/bin/activate'
DRY_RUN=0
# ------------------------------------

METHODS=( BinaryFuzzyARTMAP FuzzyARTMAP HypersphereARTMAP GaussianARTMAP )
BACKENDS=( python torch "c++" )

mkdir -p logs "$OUT_DIR"

submit_one() {
  local method="$1" backend="$2"
  local bn="${backend/c++/cpp}"
  local jobname="art-${method}-${bn}"

  local run_line=""
  if [[ -n "$LOAD_ENV_CMD" ]]; then run_line+="$LOAD_ENV_CMD; "; fi
  run_line+="python compare_factories.py --method '$method' --backend '$backend'
  --out-dir '$OUT_DIR' --seed '$SEED'"

  local args=(
    --job-name="$jobname"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --mem="$MEM_PER_JOB"
    --cpus-per-task="$CPUS_PER_TASK"
    --ntasks=1
    --output="logs/%x-%j.out"
    --error="logs/%x-%j.err"
    --export=ALL
    --parsable
    --mail-type=BEGIN,END,FAIL,REQUEUE
    --mail-user=nmmz76@umsystem.edu
  )
  [[ -n "$CONSTRAINT" ]] && args+=( --constraint="$CONSTRAINT" )

  echo "[SUBMIT] $method / $backend"
  echo "  sbatch ${args[*]} --wrap \"$run_line\""
  [[ "$DRY_RUN" -eq 0 ]] && sbatch "${args[@]}" --wrap "$run_line"
}

for m in "${METHODS[@]}"; do
  for b in "${BACKENDS[@]}"; do
    submit_one "$m" "$b"
  done
done
