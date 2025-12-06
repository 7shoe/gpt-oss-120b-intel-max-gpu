#!/bin/bash -l
#PBS -A FoundEpidem
#PBS -N test-run-container
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -l filesystems=flare:daos_user_fs
#PBS -k doe
#PBS -l place=scatter
#PBS -q debug

# Change to submission directory
cd "$PBS_O_WORKDIR"

# Number of distinct compute nodes in this job (PBS gives us PBS_NODEFILE)
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
echo "Detected $NNODES node(s) from PBS_NODEFILE"

# DAOS
module use /soft/modulefiles
module load daos

# SET
export DAOS_CONTAINER="${DAOS_CONTAINER:-samCon}"
export DAOS_POOL="${DAOS_POOL:-FoundEpidem}"

echo "Mounting ${DAOS_POOL}:${DAOS_CONTAINER} on each node"
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONTAINER}"
