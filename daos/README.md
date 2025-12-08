# Launch GPT-OSS-120B Inference across hundreds of nodes.
DAOS container for file management (parquet inputs with equations and model weights). Script transfers 

## 1. Navigate to directory
Navigate to directory that contains Vulkan, llama.cpp, and the code. 
```
cd /lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/daos
```
The key programs are 
- `run_gptoss120b_llama_server_and_daos.sh` : job submission script for debugging (`debug` queue)
- `run_gptoss120b_500.sh` : job submission script (production) 
- `infer_equations_llama_mpi.py` : Python program that launches llama servers via mpi4py
- `math_prompt.py` : contains prompt templates

## 2. Setup container
Identify your assigned DAOS pool `DAOS_POOL` and fix container name `DAOS_CONT`. The former is `FoundEpidem` for me. Check the active containers with `mount | grep dfuse`. 
```
module use /soft/modulefiles
module load daos

# check pool & container
daos pool query ${DAOS_POOL}
mount | grep dfuse

# container creation
export DAOS_POOL=FoundEpidem
export DAOS_CONT=gptOSSExprv1
daos container create --type=POSIX  --chunk-size=2097152  --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on --file-oclass=EC_16P3GX --dir-oclass=RP_4G1  ${DAOS_POOL} ${DAOS_CONT}

# - diagnose
daos pool autotest $DAOS_POOL

# - mount (& confirm)
mkdir -p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
start-dfuse.sh -m /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} --pool ${DAOS_POOL} --cont ${DAOS_CONT}
mount | grep dfuse

echo "Mounting ${DAOS_POOL}:${DAOS_CONT} on each node"
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}"
daos container create --type=POSIX  --chunk-size=2097152  --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on --file-oclass=EC_16P3GX --dir-oclass=RP_4G1  ${DAOS_POOL} ${DAOS_CONT}
```

## 3. Submit job
PBS script submits job. Define resource attributes (e.g. queue, number of compute nodes, runtime) or setup (number of llama servers per node) in the script. 
Several GPT-OSS-120B quantizations are available via GGUF format: `Q4_K_M`, `Q3_K_M`, and `Q2_K_L`. Regardless, `Q4_K_M` appears to have the best performance/accuracy tradeoff.
```
# production run
qsub run_gptoss120b_500.sh
```


## 4. Optimization
Optimization yields 4-6 llama servers per compute node. Total throughput appears maximal for 
`SERVERS_PER_NODE=6`, `N_THREADS=32` and `CTX_SIZE=1024`. With 6 XPUs per node (12 sub-devices) and 208 threads in total (192 assigned to servers) this appears reasonable. 

Below, the current per-llama server throughput:

| Metric            | Q4_K_M run      | Notes                                   |
|-------------------|-----------------|-----------------------------------------|
| prompt tok/s      | ~50 tok/s       | Q3_K_M is of same speed here            |
| generation tok/s  | ~11 tok/s       | ~2.5× slower than Q3_K_M                 |

50 tok/s for prompt processing, and 11 tok/s for generation appear reasonable for Vulkan SDK/llama.cpp setup.

## 4. Monitoring
This should look like something like 
```
aurora-pbs-XXXX.hostmgmt.cm.YYY.alcf.ZZZ.000: 
                                                                 Req'd  Req'd   Elap
Job ID               Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
-------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
1111111.aurora-pbs-* usernam* tiny     gpt120B-Q* 111111 512 10*    --  06:00 R 05:11
```
Check `gpt120B-Q4-tinyProd.o1111111` (stdout) and `gpt120B-Q4-tinyProd.e1111111` (stderr) file. After identifying a compute node address, check its log file via 
```
clush -w x4.....n0 'hostname; tail -n 50 /tmp/llama_server_0.log'
```
or the status of the files via 
```
/tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/parquet_shards
```

## 5. Unmount
Once the container has served its purpose, unmount with
```
fusermount3 -u /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
```
