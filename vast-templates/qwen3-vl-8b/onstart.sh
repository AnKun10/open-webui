#!/bin/bash
set -e

# Persist log + data dirs (do BEFORE setting tee redirect)
mkdir -p /workspace/logs /workspace/openwebui-data /workspace/.venvs /workspace/.hf_cache

exec > >(tee -a /workspace/logs/onstart.log) 2>&1
echo "=== Open WebUI bootstrap on top of vastai/vllm: $(date) ==="

# Skip if disabled by env
if [ "${OPENWEBUI_ENABLE:-true}" != "true" ]; then
  echo "OPENWEBUI_ENABLE=false → skip Open WebUI bring-up."
  exit 0
fi

# Stable JWT secret across pod restarts (so sessions survive)
if [ ! -s /workspace/.webui-secret ]; then
  openssl rand -hex 32 > /workspace/.webui-secret
fi
export WEBUI_SECRET_KEY=$(cat /workspace/.webui-secret)

# Persistent venv on /workspace
WEBUI_VENV=/workspace/.venvs/webui
if [ ! -x "$WEBUI_VENV/bin/open-webui" ]; then
  echo "[1/3] Creating Open WebUI venv (one-time, ~2-3 min)..."
  apt-get install -y python3-venv >/dev/null 2>&1 || true
  /usr/bin/python3 -m venv "$WEBUI_VENV"
  "$WEBUI_VENV/bin/pip" install --upgrade pip
  "$WEBUI_VENV/bin/pip" install open-webui aiosqlite httpx
else
  echo "[1/3] Open WebUI venv already exists, skip install."
fi

# Always ensure filter deps present (handles upgrade from older venv)
"$WEBUI_VENV/bin/pip" install --quiet aiosqlite httpx

# Start vLLM directly under tmux if nothing is already serving on :8000.
# Background: when the template runs in legacy /.launch mode (default after
# cloning a vastai/pytorch-based template), the image's supervisord chain
# never starts, so vllm.sh never runs. We launch vllm ourselves.
# If a future template uses the modern entrypoint, supervisord may already
# have started vllm — the curl probe below short-circuits us out.
if ! curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  if ! tmux has-session -t vllm 2>/dev/null; then
    echo "[2/3] vLLM not serving on :8000 — launching it under tmux..."
    tmux new -d -s vllm "\
      export HF_HOME=/workspace/.hf_cache; \
      vllm serve ${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct} \
        --host 127.0.0.1 --port 8000 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --dtype float16 \
        --served-model-name qwen3-vl-8b \
        --download-dir /workspace/.hf_cache \
      2>&1 | tee /workspace/logs/vllm.log"
  else
    echo "[2/3] vLLM tmux session already exists, leaving it alone."
  fi
else
  echo "[2/3] vLLM already serving on :8000 (image-managed), skip manual launch."
fi

# Wait for vLLM /health (image-managed or our manual tmux). Cold start can
# take 5-15 min while Qwen3-VL weights download into /workspace/.hf_cache.
echo "[3/3] Waiting for vLLM /health on :8000..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
echo "vLLM is ready. Launching Open WebUI on :3000..."

tmux kill-session -t webui 2>/dev/null || true
tmux new -d -s webui "\
  DATA_DIR=/workspace/openwebui-data \
  WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY \
  WEBUI_AUTH=True \
  ENABLE_BASE_MODELS_CACHE=false \
  OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEYS=sk-dummy \
  PORT=3000 HOST=127.0.0.1 \
  $WEBUI_VENV/bin/open-webui serve --host 127.0.0.1 --port 3000 \
  2>&1 | tee /workspace/logs/webui.log"

echo "=== Bootstrap done. SSH tunnel: ssh -L 3000:127.0.0.1:3000 ... ==="
