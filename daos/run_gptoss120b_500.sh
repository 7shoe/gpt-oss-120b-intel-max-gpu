#!/bin/bash -l
#PBS -A FoundEpidem
#PBS -N gpt120B-Q4-tinyProd
#PBS -l select=512
#PBS -l walltime=06:00:00
#PBS -l filesystems=flare:daos_user_fs
#PBS -k doe
#PBS -l place=scatter
#PBS -q prod

set -eo pipefail

echo "Job ID: ${PBS_JOBID}"
echo "Running on nodes:"
cat "${PBS_NODEFILE}"

# ----------------------------------------------------------------------------
# MODEL
# Ref.: https://huggingface.co/unsloth/gpt-oss-120b-unsloth-bnb-4bit
# Supported model variants: {Q4_K_M, Q3_K_M, Q2_K_L}
# ----------------------------------------------------------------------------
MODEL_VARIANT="Q4_K_M"
MODEL_PREFIX="gpt-oss-120b-${MODEL_VARIANT}"
MODEL_BASENAME_FIRST="${MODEL_PREFIX}-00001-of-00002.gguf"

# ----------------------------------------------------------------------------
# SERVER CONFIG (one llama-server per XPU)
# ----------------------------------------------------------------------------
BASE_PORT=18080        # first port per node; rank k uses BASE_PORT + local_rank
SERVERS_PER_NODE=6     # one per XPU
CTX_SIZE=1024          # context size for llama-server
GPU_LAYERS=90          # n-gpu-layers
N_THREADS=32           # CPU threads per server (tune later); 208 available per comp. node
LLAMA_CPP_DIR="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/scripts/llama.cpp"

# -----------------------------------------------------------------------------
# Modules / environment
# -----------------------------------------------------------------------------
module use /soft/modulefiles
module load daos
module load frameworks

# conda  (the hard-coded source is optional; can be dropped)
source /opt/aurora/25.190.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh

# Derive CONDA_BASE from "which conda":
CONDA_BIN=$(which conda)
echo "CONDA_BIN=${CONDA_BIN}"
CONDA_BASE=$(dirname "$(dirname "${CONDA_BIN}")")   # strip /condabin/conda
echo "CONDA_BASE=${CONDA_BASE}"

# Source the conda init script
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# conda env 
conda activate torch-xpu-py312

# Llama networking / XPU env
unset ONEAPI_DEVICE_SELECTOR
export FI_PROVIDER=tcp

# Make sure localhost bypasses squid
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

# navigate
cd "${PBS_O_WORKDIR}"

# -----------------------------------------------------------------------------
# DAOS: pool, container, dfuse mount on compute nodes
# -----------------------------------------------------------------------------
export DAOS_POOL=FoundEpidem
export DAOS_CONT=gptOSSExprv1

# Mount container on all compute nodes at /tmp/${DAOS_POOL}/${DAOS_CONT}
echo "Launching dfuse on all nodes..."
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}"

DAOS_ROOT="/tmp/${DAOS_POOL}/${DAOS_CONT}"

# input/output data
SRC_DIR="${DAOS_ROOT}/parquet_shards"
DST_DIR="${DAOS_ROOT}/output_tmp"

# weights (in DAOS, staged to /tmp on each node)
MODEL_DIR="${DAOS_ROOT}/models"
LOCAL_MODEL_DIR="/tmp/llama_models"
LOCAL_MODEL="${LOCAL_MODEL_DIR}/${MODEL_BASENAME_FIRST}"

# output directory on DAOS (only needs to run on one node; it's shared)
mkdir -p "${DST_DIR}"

# -----------------------------------------------------------------------------
# Transfer weights to node-local disk (mmap compatible) for each node
# -----------------------------------------------------------------------------
NODES=$(awk -F/ '{print $1}' "${PBS_NODEFILE}" | sort -u | paste -sd, -)

echo "Staging model shards from DAOS to /tmp on each node..."
clush -w "${NODES}" "bash -lc '
  set -euo pipefail
  mkdir -p ${LOCAL_MODEL_DIR}
  for src in ${MODEL_DIR}/${MODEL_PREFIX}-*.gguf; do
    base=\$(basename \"\$src\")
    dst=${LOCAL_MODEL_DIR}/\$base
    if [ ! -f \"\$dst\" ]; then
      echo \"[\$(hostname)] copying \$src -> \$dst\"
      cp \"\$src\" \"\$dst\"
    else
      echo \"[\$(hostname)] shard already present: \$dst\"
    fi
  done
'"

# this is what infer_equations_llama_mpi.py will see as --model (not used by server, but kept for API symmetry)
MODEL_PATH="${LOCAL_MODEL}"
echo "Using model (shard 1 path passed to Python): ${MODEL_PATH}"

# -----------------------------------------------------------------------------
# Start llama-server processes (one per XPU per node)
# -----------------------------------------------------------------------------
echo "Starting ${SERVERS_PER_NODE} llama-server processes per node ..."

clush -w "${NODES}" "bash -lc '
  set -euo pipefail

  LLAMA_CPP_DIR=\"${LLAMA_CPP_DIR}\"
  MODEL_PATH=\"${LOCAL_MODEL}\"
  BASE_PORT=${BASE_PORT}
  SERVERS_PER_NODE=${SERVERS_PER_NODE}
  CTX_SIZE=${CTX_SIZE}
  GPU_LAYERS=${GPU_LAYERS}
  N_THREADS=${N_THREADS}

  echo \"[\$(hostname)] launching llama-server instances...\"

  for LOCAL_RANK in \$(seq 0 \$((SERVERS_PER_NODE-1))); do
    export ONEAPI_DEVICE_SELECTOR=\"level_zero:\${LOCAL_RANK}\"
    PORT=\$((BASE_PORT + LOCAL_RANK))
    LOG=\"/tmp/llama_server_\${LOCAL_RANK}.log\"

    echo \"[\$(hostname)] XPU \${LOCAL_RANK} -> port \${PORT}, log \${LOG}\"

    \"\${LLAMA_CPP_DIR}/build/bin/llama-server\" \
      --model \"\${MODEL_PATH}\" \
      --ctx-size \"\${CTX_SIZE}\" \
      --n-gpu-layers \"\${GPU_LAYERS}\" \
      --threads \"\${N_THREADS}\" \
      --port \"\${PORT}\" \
      --host 0.0.0.0 \
      --mlock \
      >\"\${LOG}\" 2>&1 &
  done

  # Optional: record PIDs
  pgrep -f \"llama-server\" > /tmp/llama_server_pids.txt || true
'"

echo "All llama-server processes started on each node (one per XPU)."

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
echo "Waiting for llama-server processes to finish loading models on each node..."

clush -w "${NODES}" "bash -lc '
  set -euo pipefail

  # Make sure localhost doesn\'t go through squid on the remote node either
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=\${NO_PROXY}

  BASE_PORT=${BASE_PORT}
  SERVERS_PER_NODE=${SERVERS_PER_NODE}

  for LOCAL_RANK in \$(seq 0 \$((SERVERS_PER_NODE-1))); do
    PORT=\$((BASE_PORT + LOCAL_RANK))
    echo \"[\$(hostname)] Waiting for llama-server on port \${PORT}...\"

    # 1) Wait for /health to respond
    for attempt in \$(seq 1 120); do
      if curl -sS --max-time 5 \"http://127.0.0.1:\${PORT}/health\" >/dev/null 2>&1; then
        echo \"[\$(hostname)] /health OK on port \${PORT} (after \${attempt} checks)\"
        break
      fi
      sleep 5
    done

    # 2) Wait until /v1/chat/completions stops saying \"Loading model\"
    payload=\"{\\\"model\\\":\\\"gpt-oss-120b\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"ping\\\"}],\\\"max_tokens\\\":1}\"

    for attempt in \$(seq 1 120); do
      resp=\$(curl -sS --max-time 30 -X POST \
        -H \"Content-Type: application/json\" \
        --data \"\${payload}\" \
        \"http://127.0.0.1:\${PORT}/v1/chat/completions\" || true)

      if echo \"\${resp}\" | grep -q \"Loading model\"; then
        echo \"[\$(hostname)] port \${PORT}: model still loading (attempt \${attempt})\"
        sleep 10
        continue
      fi

      # If we got here and resp is non-empty, assume model is ready
      echo \"[\$(hostname)] port \${PORT}: model ready.\"
      break
    done
  done
'"

echo "All llama-server instances report ready; starting MPI clients..."


# -----------------------------------------------------------------------------
# Lustre destination (long-term storage of results)
# -----------------------------------------------------------------------------
LUSTRE_OUT="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/daos_output"
mkdir -p "${LUSTRE_OUT}"

echo "DAOS_ROOT       = ${DAOS_ROOT}"
echo "SRC_DIR         = ${SRC_DIR}"
echo "DST_DIR         = ${DST_DIR}"
echo "LOCAL_MODEL_DIR = ${LOCAL_MODEL_DIR}"
echo "LUSTRE_OUT      = ${LUSTRE_OUT}"
echo "BASE_PORT       = ${BASE_PORT}"
echo "SERVERS_PER_NODE= ${SERVERS_PER_NODE}"

# -----------------------------------------------------------------------------
# Periodic rsync from DAOS -> Lustre (runs on the job launch node)
# -----------------------------------------------------------------------------
rsync_loop() {
  while true; do
    echo "[RSYNC] $(date): syncing DAOS -> Lustre (outputs only, no checkpoints)..."
    rsync -a --update --partial --inplace \
      --exclude 'checkpoints/' \
      --exclude '*.ckpt.json' \
      "${DST_DIR}/" \
      "${LUSTRE_OUT}/"
    echo "[RSYNC] $(date): sync done."
    sleep 1600  # 27 minutes
  done
}

# Start rsync loop in the background
rsync_loop &
RSYNC_PID=$!
echo "Started background rsync loop with PID ${RSYNC_PID}"

# -----------------------------------------------------------------------------
# MPI launch configuration
# -----------------------------------------------------------------------------
NNODES=$(wc -l < "${PBS_NODEFILE}")
RANKS_PER_NODE=${SERVERS_PER_NODE}   # one rank per server/XPU
NRANKS=$(( NNODES * RANKS_PER_NODE ))

echo "NNODES        = ${NNODES}"
echo "RANKS_PER_NODE= ${RANKS_PER_NODE}"
echo "NRANKS        = ${NRANKS}"

# -----------------------------------------------------------------------------
# Launch the MPI-driven inference
# -----------------------------------------------------------------------------
echo "Starting MPI inference at: $(date)"

mpiexec -np "${NRANKS}" -ppn "${RANKS_PER_NODE}" \
  python infer_equations_llama_mpi.py \
    --src "${SRC_DIR}" \
    --dst "${DST_DIR}" \
    --model "${MODEL_PATH}" \
    --ctx "${CTX_SIZE}" \
    --ngl "${GPU_LAYERS}"

MPI_STATUS=$?

echo "MPI finished at: $(date) with status ${MPI_STATUS}"

# -----------------------------------------------------------------------------
# Final sync + cleanup
# -----------------------------------------------------------------------------
echo "Final rsync before exit..."
rsync -a --update --partial --inplace \
  "${DST_DIR}/" \
  "${LUSTRE_OUT}/"

# Stop background rsync loop
kill "${RSYNC_PID}" 2>/dev/null || true

# Stop llama-server processes on all nodes
echo "Stopping llama-server processes on all nodes..."
clush -w "${NODES}" "bash -lc '
  pkill -f llama-server || true
  rm -f /tmp/llama_server_pids.txt
'"

echo "Unmounting dfuse..."
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" || true

echo "Job done."
exit "${MPI_STATUS}"

