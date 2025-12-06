#!/usr/bin/env python
from mpi4py import MPI
import os
import socket

# ----- MPI setup -----
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

# Node-local communicator to compute local rank per node
node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
local_rank = node_comm.Get_rank()

hostname = socket.gethostname()

# Bind this rank to one Intel XPU (Level Zero device = local_rank)
os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{local_rank}"

print(
    f"[WORLD {world_rank}/{world_size} | NODE {hostname} | LOCAL {local_rank}] "
    f"ONEAPI_DEVICE_SELECTOR={os.environ['ONEAPI_DEVICE_SELECTOR']}",
    flush=True,
)

# Optional barrier so output from all ranks is flushed before exit
comm.Barrier()

