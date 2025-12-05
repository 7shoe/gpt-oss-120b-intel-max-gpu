#!/bin/bash
set -euo pipefail

echo "=== Downloading GPT-OSS-120B Q4_K_M GGUF shards ==="

MODEL_DIR="../models"
mkdir -p "$MODEL_DIR"

cd "$MODEL_DIR"

echo "Downloading shard 1..."
wget -O gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  "https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

echo "Downloading shard 2..."
wget -O gpt-oss-120b-Q4_K_M-00002-of-00002.gguf \
  "https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/Q4_K_M/gpt-oss-120b-Q4_K_M-00002-of-00002.gguf"

echo "=== Download complete ==="
ls -lh gpt-oss-120b-Q4_K_M-*.gguf

