#!/bin/bash
# run_inference.sh
unset ONEAPI_DEVICE_SELECTOR
export FI_PROVIDER=tcp

mpirun -n 24 python infer_equations_llama_mpi.py \
    --src "/lus/flare/projects/FoundEpidem/siebenschuh/DocDiffuser/data/expressions/gptoss120b_SMALL_input/" \
    --dst "/lus/flare/projects/FoundEpidem/siebenschuh/DocDiffuser/data/expressions/gptoss120b_NEW_SMALL_inferred/" \
    --model "/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/models/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf" \
    --ctx 1024
