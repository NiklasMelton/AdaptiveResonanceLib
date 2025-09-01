#!/usr/bin/env bash
set -euo pipefail

# -------- locate this launcher & the python script ----------
# Absolute path to the directory that contains THIS launcher
# Works with sourced, symlinked, or directly executed scripts.
SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
script="${SCRIPT_DIR}/compare_factories.py"
# -----------------------------------------------------------

# -------- child job config ----------
PARTITION="general"
CONSTRAINT="intel&skylake&40CPU"   # choose one: amd&64CPU | amd&128CPU | intel&skylake&40CPU
TIME_LIMIT="0-12:00:00"
MEM_PER_JOB="16G"
CPUS_PER_TASK=8
SEED=0
OUT_DIR="results"                  # outputs relative to *current* working dir
LOAD_ENV_CMD="module load python/3.12.1"   # or: source /path/to/venv/bin/activate
DRY_RUN=0
# ------------------------------------

METHODS=( BinaryFuzzyARTMAP FuzzyARTMAP HypersphereARTMAP GaussianARTMAP )
BACKENDS=( python torch "c++" )

mkdir -p logs "$OUT_DIR"

submit_one() {
  local method="$1"
  local backend="$2"

  local bn="${backend/c++/cpp}"
  local jobname="art-${method}-${bn}"

  # Build python command safely (single line, no embedded newlines)
  local -a py_args
  py_args+=( --method "$method" )
  py_args+=( --backend "$backend" )
  py_args+=( --out-dir "$OUT_DIR" )
  py_args+=( --seed "$SEED" )

  local child_cmd="python3 $script"
  for a in "${py_args[@]}"; do
    child_cmd+=" $(printf '%q' "$a")"
  done

  if [[ -n "$LOAD_ENV_CMD" ]]; then
    child_cmd="$LOAD_ENV_CMD; $child_cmd"
  fi

  # Run under bash -lc so 'module' / venv activation works
  local wrap_cmd="bash -lc $(printf '%q' "$child_cmd")"

  # sbatch args for the child job
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
  echo "  sbatch ${args[*]} --wrap $wrap_cmd"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    sbatch "${args[@]}" --wrap "$wrap_cmd"
  fi
}

for m in "${METHODS[@]}"; do
  for b in "${BACKENDS[@]}"; do
    submit_one "$m" "$b"
  done
done
