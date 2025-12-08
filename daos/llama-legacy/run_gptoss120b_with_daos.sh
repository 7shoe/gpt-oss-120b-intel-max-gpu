#!/bin/bash -l
#PBS -A FoundEpidem
#PBS -N llama-daos-infer
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:daos_user_fs
#PBS -k doe
#PBS -l place=scatter
#PBS -q debug

set -eo pipefail

echo "Job ID: ${PBS_JOBID}"
echo "Running on nodes:"
cat "${PBS_NODEFILE}"

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
MODEL_PREFIX="gpt-oss-120b-Q4_K_M"
MODEL_BASENAME_FIRST="${MODEL_PREFIX}-00001-of-00002.gguf"
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

# weights
MODEL_DIR="${DAOS_ROOT}/models"
LOCAL_MODEL_DIR="/tmp/llama_models"
LOCAL_MODEL="${LOCAL_MODEL_DIR}/${MODEL_BASENAME_FIRST}"

# output directory on DAOS (only needs to run on one node; it's shared)
mkdir -p "${DST_DIR}"

# -----------------------------------------------------------------------------
# Transfer weights to node-local disk (mmap compatible) for each node
# -----------------------------------------------------------------------------
NODES=$(awk -F/ '{print $1}' "${PBS_NODEFILE}" | sort -u | paste -sd, -)

clush -w "${NODES}" "
  mkdir -p ${LOCAL_MODEL_DIR} ;
  for src in ${MODEL_DIR}/${MODEL_PREFIX}-*.gguf; do
    base=\$(basename \"\$src\")
    dst=${LOCAL_MODEL_DIR}/\$base
    if [ ! -f \"\$dst\" ]; then
      echo \"[\$(hostname)] copying \$src -> \$dst\";
      cp \"\$src\" \"\$dst\";
    else
      echo \"[\$(hostname)] shard already present: \$dst\";
    fi
  done
"

# this is what infer_equations_llama_mpi.py will see as --model
MODEL_PATH="${LOCAL_MODEL}"
echo "Using model: ${MODEL_PATH}"

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

# -----------------------------------------------------------------------------
# Launch the MPI-driven inference
# -----------------------------------------------------------------------------
echo "Starting MPI inference at: $(date)"

mpiexec -np "${NRANKS}" -ppn "${RANKS_PER_NODE}" \
  python infer_equations_llama_mpi.py \
    --src "${SRC_DIR}" \
    --dst "${DST_DIR}" \
    --model "${MODEL_PATH}" \
    --ctx 1536 \
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

