#!/bin/bash -l
#PBS -A FoundEpidem
#PBS -N llama-daos-infer
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:daos_user_fs
#PBS -k doe
#PBS -l place=scatter
#PBS -q debug

set -euo pipefail

echo "Job ID: ${PBS_JOBID}"
echo "Running on nodes:"
cat "${PBS_NODEFILE}"

# -----------------------------------------------------------------------------
# Modules / environment
# -----------------------------------------------------------------------------
module use /soft/modulefiles
module load daos
module load frameworks

# conda 
source /opt/aurora/25.190.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh

# Derive CONDA_BASE from "which conda":
CONDA_BIN=$(which conda)
echo "CONDA_BIN=${CONDA_BIN}"
CONDA_BASE=$(dirname "$(dirname "${CONDA_BIN}")")   # strip /condabin/conda
echo "CONDA_BASE=${CONDA_BASE}"

# Source the conda init script
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate torch-xpu-py312


# Llama networking / XPU env
unset ONEAPI_DEVICE_SELECTOR
export FI_PROVIDER=tcp

# naviagte
cd $PBS_O_WORKDIR

# -----------------------------------------------------------------------------
# DAOS: pool, container, dfuse mount on compute nodes
# -----------------------------------------------------------------------------
export DAOS_POOL=FoundEpidem
export DAOS_CONT=gptOSSExprv1

# Mount container on all compute nodes at /tmp/${DAOS_POOL}/${DAOS_CONT}
echo "Launching dfuse on all nodes..."
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}"

DAOS_ROOT="/tmp/${DAOS_POOL}/${DAOS_CONT}"
SRC_DIR="${DAOS_ROOT}/parquet_shards"
DST_DIR="${DAOS_ROOT}/output_tmp"
MODEL_DIR="${DAOS_ROOT}/models"

# Create output directory on DAOS (only needs to run on one node; it's shared)
mkdir -p "${DST_DIR}"

# -----------------------------------------------------------------------------
# Lustre destination (long-term storage of results)
# -----------------------------------------------------------------------------
LUSTRE_OUT="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/daos_output"
mkdir -p "${LUSTRE_OUT}"

echo "DAOS_ROOT   = ${DAOS_ROOT}"
echo "SRC_DIR     = ${SRC_DIR}"
echo "DST_DIR     = ${DST_DIR}"
echo "MODEL_DIR   = ${MODEL_DIR}"
echo "LUSTRE_OUT  = ${LUSTRE_OUT}"

# -----------------------------------------------------------------------------
# Optional: stage model to node-local /tmp once per node
# (You can skip this and just use MODEL_PATH on DAOS directly.)
# -----------------------------------------------------------------------------
# MODEL_ON_DAOS="${MODEL_DIR}/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
# MODEL_LOCAL="/tmp/llama-model.gguf"
# echo "Staging model to /tmp on each node (optional)..."
# clush --hostfile "${PBS_NODEFILE}" "if [ ! -f ${MODEL_LOCAL} ]; then cp ${MODEL_ON_DAOS} ${MODEL_LOCAL}; fi"
# MODEL_PATH="${MODEL_LOCAL}"

# Simpler: read model directly from DAOS
MODEL_PATH="${MODEL_DIR}/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

echo "Using model: ${MODEL_PATH}"

# -----------------------------------------------------------------------------
# Periodic rsync from DAOS -> Lustre (runs on the job launch node)
# -----------------------------------------------------------------------------
rsync_loop() {
  while true; do
    echo "[RSYNC] $(date): syncing DAOS -> Lustre..."
    # Only copy new/updated files; avoid re-writing unchanged files.
    rsync -a --update --partial --inplace \
      "${DST_DIR}/" \
      "${LUSTRE_OUT}/"
    echo "[RSYNC] $(date): sync done."
    sleep 1800  # 30 minutes
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
RANKS_PER_NODE=12           # one rank per XPU
NRANKS=$(( NNODES * RANKS_PER_NODE ))

echo "NNODES        = ${NNODES}"
echo "RANKS_PER_NODE= ${RANKS_PER_NODE}"
echo "NRANKS        = ${NRANKS}"

# If you want explicit CPU binding, you can add --cpu-bind or reuse Aurora examples.

# -----------------------------------------------------------------------------
# Launch the MPI-driven inference
# -----------------------------------------------------------------------------
echo "Starting MPI inference at: $(date)"

mpiexec -np "${NRANKS}" -ppn "${RANKS_PER_NODE}" \
  --no-vni \
  python infer_equations_llama_mpi.py \
    --src "${SRC_DIR}" \
    --dst "${DST_DIR}" \
    --model "${MODEL_PATH}" \
    --ctx 1024 \
    --ngl 80

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

echo "Unmounting dfuse..."
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" || true

echo "Job done."
exit "${MPI_STATUS}"

