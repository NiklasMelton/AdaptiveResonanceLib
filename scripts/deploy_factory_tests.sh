#!/usr/bin/env bash
set -euo pipefail

# ───────── Config ─────────
# Paths to the two job templates (assumed to live next to this launcher)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TPL_CPU="${SCRIPT_DIR}/run_template.slurm"
TPL_GPU="${SCRIPT_DIR}/run_template_gpu.slurm"

# Common parameters you can tweak
SEED=0
MAKE_LOGS_DIR=1       # create logs/ if missing
DRY_RUN=0             # 1 = print sbatch lines only, 0 = actually submit

# Leave blank to rely on the #SBATCH lines inside each template.
PARTITION_CPU="gpu"
CONSTRAINT_CPU="intel&skylake&40CPU"
PARTITION_GPU="gpu"
CONSTRAINT_GPU=""
GRES_GPU="gpu:V100-SXM2-32GB:1"
# ─────────────────────────

METHODS=( BinaryFuzzyARTMAP FuzzyARTMAP HypersphereARTMAP GaussianARTMAP )
BACKENDS=( python torch "c++" )

(( MAKE_LOGS_DIR )) && mkdir -p "${SCRIPT_DIR}/logs"

submit_cpu() {
  local method="$1" backend="$2"
  local bn="${backend/c++/cpp}"
  local jobname="ART-${method}-${bn}-cpu"

  local args=(
    --job-name="${jobname}"
    --output="logs/%x-%j.out"
    --error="logs/%x-%j.err"
    --export=ALL,METHOD="${method}",BACKEND="${backend}",DEVICE=cpu,SEED="${SEED}"
  )
  [[ -n "${PARTITION_CPU}"  ]] && args+=( --partition="${PARTITION_CPU}" )
  [[ -n "${CONSTRAINT_CPU}" ]] && args+=( --constraint="${CONSTRAINT_CPU}" )

  echo "[CPU]  sbatch ${args[*]} ${TPL_CPU}"
  (( DRY_RUN )) || sbatch "${args[@]}" "${TPL_CPU}"
}

submit_gpu() {
  local method="$1"                # backend is torch here
  local jobname="ART-${method}-torch-gpu"
  local LOGDIR="$HOME/logs"
  mkdir -p "$LOGDIR"

  local args=(
    --job-name="$jobname"
    --chdir="$HOME"                                         # <- known work dir
    --output="$LOGDIR/%x-%j.out"                            # <- absolute logs
    --error="$LOGDIR/%x-%j.err"
    --partition=gpu
    --constraint='intel&skylake&40CPU'                      # <- QUOTED
    --gres=gpu:V100-SXM2-32GB:1
    --export=ALL,METHOD="$method",BACKEND=torch,DEVICE=gpu,SEED="$SEED"
    --parsable
  )

  echo "[GPU] sbatch ${args[*]} $TPL_GPU"
  jid=$(sbatch "${args[@]}" "$TPL_GPU") || { echo "ERROR: sbatch failed"; return 1; }
  echo "[GPU] Submitted JobID=$jid"
  scontrol show job "$jid" | egrep 'WorkDir|StdOut|StdErr|Partition|Gres|Reason'
}


# ───────── Submit 16 jobs ─────────
for method in "${METHODS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    # CPU job for every (method, backend)
    submit_cpu "$method" "$backend"
    # GPU job only for torch
    if [[ "$backend" == "torch" ]]; then
      submit_gpu "$method"
    fi
  done
done
