#!/usr/bin/env bash
set -euo pipefail

# ── locate this launcher & the python script (works no matter where you call it) ──
SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
SCRIPT_PY="${SCRIPT_DIR}/compare_factories.py"

# -------- child job config ----------
PARTITION="general"
CONSTRAINT="intel&skylake&40CPU"   # choose: amd&64CPU | amd&128CPU | intel&skylake&40CPU
TIME_LIMIT="0-12:00:00"
MEM_PER_JOB="16G"
CPUS_PER_TASK=8
SEED=0

# Put logs/results next to this launcher (absolute paths avoid missing-log issues)
LOG_DIR="${SCRIPT_DIR}/logs"
OUT_DIR="${SCRIPT_DIR}/results"

# Where child jobs should start (project root)
CHDIR="${SCRIPT_DIR}"

# Load Python (module or venv). If empty, nothing is loaded.
LOAD_ENV_CMD="module load python/3.12.1"

# Dry run = print sbatch lines but don’t submit (0|1)
DRY_RUN=0
# ------------------------------------

METHODS=( BinaryFuzzyARTMAP FuzzyARTMAP HypersphereARTMAP GaussianARTMAP )
BACKENDS=( python torch "c++" )

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

submit_one() {
  local method="$1"
  local backend="$2"

  local bn="${backend/c++/cpp}"
  local jobname="art-${method}-${bn}"

  # Build python command safely (single line, shell-escaped)
  local -a py_args
  py_args+=( --method "$method" )
  py_args+=( --backend "$backend" )
  py_args+=( --out-dir "$OUT_DIR" )
  py_args+=( --seed "$SEED" )

  local child_cmd="python3 ${SCRIPT_PY}"
  for a in "${py_args[@]}"; do
    child_cmd+=" $(printf '%q' "$a")"
  done

  # Compose what runs on the worker. Use a *login* Bash (-l) so environment modules exist.
  # Add tracing so failures show up in logs.
  local preface="set -euxo pipefail; echo '[NODE]' \"\$(hostname)\"; echo '[PWD]' \"\$(pwd)\";"
  if [[ -n "$LOAD_ENV_CMD" ]]; then
    preface+=" ${LOAD_ENV_CMD}; which python3; python3 --version;"
  fi
  local wrap_cmd="bash -l -c $(printf '%q' "${preface} ${child_cmd}")"

  # sbatch args for the child job
  local args=(
    --job-name="$jobname"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --mem="$MEM_PER_JOB"
    --cpus-per-task="$CPUS_PER_TASK"
    --ntasks=1
    --chdir="$CHDIR"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    --export=ALL
    --parsable
  )
  [[ -n "$CONSTRAINT" ]] && args+=( --constraint="$CONSTRAINT" )
  # If your site uses Slurm's user env import, uncomment:
  # args+=( --get-user-env=L )

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
