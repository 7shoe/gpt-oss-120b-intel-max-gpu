

### 1. Input parquets
16 original parquet files with 160M equations split into ~5K files (32K rows each). 
```
/lus/flare/projects/FoundEpidem/siebenschuh/DocDiffuser/data/expressions/split_all_parquets_into_5000_files
```

### 2. DAOS Container
Set up the container and transfer the input parquets:
```
module use /soft/modulefiles
module load daos

export DAOS_POOL=FoundEpidem
export DAOS_CONT=gptOSSExprv1
daos container create --type=POSIX  --chunk-size=2097152  --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on --file-oclass=EC_16P3GX --dir-oclass=RP_4G1  ${DAOS_POOL} ${DAOS_CONT}

daos pool query ${DAOS_POOL}

mkdir -p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
mkdir -p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/parquet_shards/
start-dfuse.sh -m /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} --pool ${DAOS_POOL} --cont ${DAOS_CONT}

# transfer parquets
daos filesystem copy \
  --src /lus/flare/projects/FoundEpidem/siebenschuh/DocDiffuser/data/expressions/split_all_parquets_into_5000_files \
  --dst daos://${DAOS_POOL}/${DAOS_CONT}/parquet_shards

# transfer model weights
mkdir -p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/models
cp /lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/models/*.gguf \
   /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/models/

# create destination dir
mkdir -p /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/output_tmp
```

### 3. Launch 
Navigate to the directory and launch 
```
cd /lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/daos
qsub run_gptoss120b_with_daos.sh
```
