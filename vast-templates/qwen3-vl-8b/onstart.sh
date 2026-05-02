#!/bin/bash
set -e
exec > >(tee -a /workspace/logs/onstart.log) 2>&1

echo "=== Open WebUI + vLLM bootstrap: $(date) ==="

mkdir -p /workspace/logs /workspace/.hf_cache /workspace/.venvs /workspace/openwebui-data
export HF_HOME=/workspace/.hf_cache
export PIP_ROOT_USER_ACTION=ignore
SYS_PY=/usr/bin/python3

# NEW (filter persistence): persistent data dir for Open WebUI DB + caption cache
export DATA_DIR=/workspace/openwebui-data

# NEW (filter persistence): stable JWT secret across pod restarts
export WEBUI_SECRET_KEY=$(cat /workspace/.webui-secret 2>/dev/null \
    || (openssl rand -hex 32 | tee /workspace/.webui-secret))

# ---- 1. vLLM in system Python (idempotent) ----
if ! $SYS_PY -c "import vllm" 2>/dev/null; then
  echo "[1/4] Installing vllm..."
  $SYS_PY -m pip install --no-cache-dir \
    vllm==0.19.1 \
    "transformers!=5.3.*" \
    "accelerate==1.12.0" \
    "aiohttp>=3.13.3"
else
  echo "[1/4] vllm already installed, skip."
fi

# ---- 2. Open WebUI in ISOLATED venv ----
WEBUI_VENV=/workspace/.venvs/webui
if [ ! -x "$WEBUI_VENV/bin/open-webui" ]; then
  echo "[2/4] Creating isolated venv for open-webui..."
  apt-get install -y python3-venv >/dev/null 2>&1 || true
  $SYS_PY -m venv "$WEBUI_VENV"
  "$WEBUI_VENV/bin/pip" install --no-cache-dir --upgrade pip
  "$WEBUI_VENV/bin/pip" install --no-cache-dir open-webui aiosqlite httpx
else
  echo "[2/4] open-webui venv exists, skip."
fi

# ---- 3. Start vLLM (port 8000) ----
echo "[3/4] Starting vLLM..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new -d -s vllm "\
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  HF_HOME=/workspace/.hf_cache \
  /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name qwen3-vl-8b \
    --host 127.0.0.1 --port 8000 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --mm-encoder-attn-backend TORCH_SDPA \
    2>&1 | tee /workspace/logs/vllm.log"

# ---- 4. Start Open WebUI (port 3000) ----
echo "[4/4] Waiting for vLLM readiness, then starting Open WebUI..."
tmux kill-session -t webui 2>/dev/null || true
tmux new -d -s webui "\
  until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done; \
  echo 'vLLM ready, launching Open WebUI...'; \
  ENABLE_BASE_MODELS_CACHE=false \
  DATA_DIR=$DATA_DIR \
  WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY \
  OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEYS=sk-dummy \
  PORT=3000 HOST=127.0.0.1 \
  WEBUI_AUTH=True \
  $WEBUI_VENV/bin/open-webui serve --host 127.0.0.1 --port 3000 \
  2>&1 | tee /workspace/logs/webui.log"

echo "=== Bootstrap done. Tail logs: tmux attach -t vllm | webui ==="
