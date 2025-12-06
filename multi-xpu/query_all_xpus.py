#!/usr/bin/env python3
"""
Query all XPUs across all nodes in an MPI allocation.
Each rank binds to one XPU (level_zero:<local_rank>) and reports device info.

Use:
    unset ONEAPI_DEVICE_SELECTOR
    export FI_PROVIDER=tcp
    mpirun -n 24 python query_all_xpus.py
"""

from mpi4py import MPI
import os
import socket
import subprocess


# -----------------------------------------------------------------------------
# MPI Setup
# -----------------------------------------------------------------------------
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

# Get node-local communicator so each node computes its own local rank
node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
local_rank = node_comm.Get_rank()

hostname = socket.gethostname()

# Bind this rank to exactly one XPU device on the node
os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{local_rank}"

# -----------------------------------------------------------------------------
# Query device using sycl-ls restricted by ONEAPI_DEVICE_SELECTOR
# -----------------------------------------------------------------------------
def query_xpu() -> str:
    """Return a one-line sycl-ls description of the XPU assigned."""
    try:
        out = subprocess.check_output(
            ["sycl-ls"],
            stderr=subprocess.STDOUT,
            text=True
        )
        # sycl-ls prints multiple lines, but with ONEAPI_DEVICE_SELECTOR set,
        # it prints only the bound device.
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"sycl-ls error: {e.output}"


device_info = query_xpu()

# -----------------------------------------------------------------------------
# Print summary for this rank
# -----------------------------------------------------------------------------
print(
    f"[WORLD {world_rank:02d}/{world_size:02d}] "
    f"NODE={hostname} LOCAL_RANK={local_rank:02d} "
    f"XPU=level_zero:{local_rank}\n"
    f"{device_info}\n",
    flush=True
)

comm.Barrier()

