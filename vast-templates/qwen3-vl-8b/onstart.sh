#!/bin/bash
set -e

# Persist log + data dirs (do BEFORE setting tee redirect)
mkdir -p /workspace/logs /workspace/openwebui-data /workspace/.venvs

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
  echo "[1/2] Creating Open WebUI venv (one-time, ~2-3 min)..."
  apt-get install -y python3-venv >/dev/null 2>&1 || true
  /usr/bin/python3 -m venv "$WEBUI_VENV"
  "$WEBUI_VENV/bin/pip" install --upgrade pip
  "$WEBUI_VENV/bin/pip" install open-webui aiosqlite httpx
else
  echo "[1/2] Open WebUI venv already exists, skip install."
fi

# Always ensure filter deps present (handles upgrade from older venv)
"$WEBUI_VENV/bin/pip" install --quiet aiosqlite httpx

# Wait for vLLM (managed by image's supervisord) to be reachable
echo "[2/2] Waiting for image-managed vLLM to be ready on :8000..."
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
