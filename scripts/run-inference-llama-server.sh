#!/bin/bash
# MVP: start llama-server for GPT-OSS-120B and send a few test prompts.

set -euo pipefail

# -------------------------------
# Model variant / config knobs
# -------------------------------
MODEL_VARIANT="Q4_K_M"
#MODEL_VARIANT="Q3_K_M"
#MODEL_VARIANT="Q2_K_L"

MODEL_PREFIX="gpt-oss-120b-${MODEL_VARIANT}"

LLAMA_CPP_DIR="llama.cpp"
MODEL_SHARD1="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/models/${MODEL_PREFIX}-00001-of-00002.gguf"
MODEL_SHARD2="/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/models/${MODEL_PREFIX}-00002-of-00002.gguf"

CTX_SIZE=1024           # we keep this small-ish
GPU_LAYERS=80           # offload 80 layers to XPU
N_THREADS=32            # fewer CPU threads to avoid oversubscription
SERVER_PORT=8080
WAIT_SEC_BETWEEN_PINGS=5
MAX_PINGS=60            # up to 5 minutes of "loading" tolerance

echo "=== Starting llama-server for GPT-OSS-120B (${MODEL_VARIANT}) ==="
echo "Model shard 1: ${MODEL_SHARD1}"
echo "Model shard 2: ${MODEL_SHARD2}"
echo "CTX_SIZE    : ${CTX_SIZE}"
echo "GPU_LAYERS  : ${GPU_LAYERS}"
echo "N_THREADS   : ${N_THREADS}"
echo "SERVER_PORT : ${SERVER_PORT}"
echo

# Sanity checks ---------------------------------------------------------------
if [ ! -f "${MODEL_SHARD1}" ]; then
    echo "ERROR: Missing shard 1 at ${MODEL_SHARD1}"
    exit 1
fi
if [ ! -f "${MODEL_SHARD2}" ]; then
    echo "ERROR: Missing shard 2 at ${MODEL_SHARD2}"
    exit 1
fi
if [ ! -x "${LLAMA_CPP_DIR}/build/bin/llama-server" ]; then
    echo "ERROR: ${LLAMA_CPP_DIR}/build/bin/llama-server not found or not executable"
    exit 1
fi

# Intel oneAPI (if not already sourced) --------------------------------------
if [ -z "${ONEAPI_ROOT:-}" ]; then
    if [ -f "/opt/aurora/25.190.0/oneapi/setvars.sh" ]; then
        echo "Sourcing Intel oneAPI environment..."
        # shellcheck disable=SC1091
        source /opt/aurora/25.190.0/oneapi/setvars.sh
    else
        echo "WARNING: ONEAPI_ROOT not set and setvars.sh not found. Continuing anyway."
    fi
fi

# Ensure proxy does *not* affect local 127.0.0.1 traffic ----------------------
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

# Select XPU 0 explicitly
export ONEAPI_DEVICE_SELECTOR="level_zero:0"

# -----------------------------------------------------------------------------
# Start llama-server in the background
# -----------------------------------------------------------------------------
cd "${LLAMA_CPP_DIR}"

LOG_FILE="server.log"
rm -f "${LOG_FILE}"

# Note: only shard1 is given; llama.cpp auto-discovers shard2 by naming pattern.
./build/bin/llama-server \
    -m "${MODEL_SHARD1}" \
    --port "${SERVER_PORT}" \
    --ctx-size "${CTX_SIZE}" \
    -ngl "${GPU_LAYERS}" \
    -t "${N_THREADS}" \
    --log-disable \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "llama-server PID: ${SERVER_PID}"

# Simple guard to bail out early if server crashes immediately
sleep 2
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: llama-server exited immediately. Last log lines:"
    tail -n 50 "${LOG_FILE}" || true
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase 1: wait for HTTP port to listen (/health)
# -----------------------------------------------------------------------------
echo "Waiting for server /health endpoint to respond..."
for i in $(seq 1 30); do
    if curl -sS -x "" --max-time 2 "http://127.0.0.1:${SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "HTTP /health is reachable (after ${i} checks)."
        break
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: llama-server died while waiting for /health."
        tail -n 50 "${LOG_FILE}" || true
        exit 1
    fi

    sleep 1
    if [ "${i}" -eq 30 ]; then
        echo "ERROR: /health did not respond in time."
        tail -n 50 "${LOG_FILE}" || true
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Phase 2: wait until model is actually loaded (no more 503 Loading model)
# -----------------------------------------------------------------------------
echo "Waiting for model to finish loading (watching for 503 -> OK)..."
PING_PROMPT="ping"
PING_BODY=$(cat <<EOF
{
  "model": "gpt-oss-120b",
  "messages": [
    {"role": "user", "content": "${PING_PROMPT}"}
  ],
  "max_tokens": 1,
  "temperature": 0.0
}
EOF
)

for i in $(seq 1 "${MAX_PINGS}"); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: llama-server died while loading model."
        tail -n 50 "${LOG_FILE}" || true
        exit 1
    fi

    RESP=$(curl -sS -x "" \
        -H "Content-Type: application/json" \
        -d "${PING_BODY}" \
        "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" || true)

    if echo "${RESP}" | grep -q '"Loading model"'; then
        echo "[${i}/${MAX_PINGS}] Still loading model..."
    else
        echo "Model appears ready (no 'Loading model' in response)."
        break
    fi

    if [ "${i}" -eq "${MAX_PINGS}" ]; then
        echo "ERROR: model still not ready after ${MAX_PINGS} pings."
        echo "Last response:"
        echo "${RESP}"
        tail -n 50 "${LOG_FILE}" || true
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi

    sleep "${WAIT_SEC_BETWEEN_PINGS}"
done

# -----------------------------------------------------------------------------
# Send a few real test prompts
# -----------------------------------------------------------------------------
echo
echo "=== Sending test prompts to llama-server ==="
PROMPTS=(
  "What is 2+2?"
  "State the definition of a group in abstract algebra."
  "List three prime numbers greater than 10."
)

for P in "${PROMPTS[@]}"; do
    echo
    echo ">>> PROMPT:"
    echo "${P}"
    echo

    REQ=$(cat <<EOF
{
  "model": "gpt-oss-120b",
  "messages": [
    {"role": "user", "content": "${P}"}
  ],
  "max_tokens": 128,
  "temperature": 0.1
}
EOF
)
    RESP=$(curl -sS -x "" \
        -H "Content-Type: application/json" \
        -d "${REQ}" \
        "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions")
    echo "${RESP}"
    echo "<<< END RESPONSE"
    echo "------------------------------------------------------------"
done

echo
echo "=== Done with test prompts ==="

# -----------------------------------------------------------------------------
# Teardown
# -----------------------------------------------------------------------------
echo "Stopping llama-server (PID: ${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
sleep 2
if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "llama-server still alive; sending SIGKILL..."
    kill -9 "${SERVER_PID}" 2>/dev/null || true
fi

echo "=== Server stopped. ==="

