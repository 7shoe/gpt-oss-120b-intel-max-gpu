#!/bin/bash
# Run inference with GPT-OSS-120B on Intel MAX GPU 1550

set -e

# Configuration
LLAMA_CPP_DIR="llama.cpp"
MODEL_PATH="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/models/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
DEFAULT_PROMPT="What is 2+2?"
GPU_LAYERS=80  # Offload 80 layers to GPU
CONTEXT_SIZE=512
MAX_TOKENS=200

# Parse command line arguments
PROMPT="${1:-$DEFAULT_PROMPT}"

echo "=== Running GPT-OSS-120B inference on Intel MAX GPU 1550 ==="

# Check if Intel oneAPI is available
if [ -z "$ONEAPI_ROOT" ]; then
    if [ -f "/opt/aurora/25.190.0/oneapi/setvars.sh" ]; then
        echo "Sourcing Intel oneAPI environment..."
        source /opt/aurora/25.190.0/oneapi/setvars.sh
    else
        echo "ERROR: Intel oneAPI not found."
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Run ./download-gpt-oss-120b.sh first"
    exit 1
fi

# Check if second part exists
MODEL_PATH2="${MODEL_PATH/-00001-of-00002.gguf/-00002-of-00002.gguf}"
if [ ! -f "$MODEL_PATH2" ]; then
    echo "ERROR: Model part 2 not found at $MODEL_PATH2"
    echo "Both model files are required for GPT-OSS-120B"
    exit 1
fi

# Check if llama-cli exists
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-cli" ]; then
    echo "ERROR: llama-cli not found. Run ./build-llama-cpp.sh first"
    exit 1
fi

# Set GPU device selector
export ONEAPI_DEVICE_SELECTOR="level_zero:0"

cd "$LLAMA_CPP_DIR"

echo "Model: $MODEL_PATH (+ part 2)"
echo "Prompt: $PROMPT"
echo "GPU Layers: $GPU_LAYERS"
echo "Context Size: $CONTEXT_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo ""
echo "Note: Loading 120B model takes ~60-90 seconds..."
echo ""

# Run inference
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -ngl $GPU_LAYERS \
    -c $CONTEXT_SIZE \
    -n $MAX_TOKENS

echo ""
echo "=== Inference complete ==="
