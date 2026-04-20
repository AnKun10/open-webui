#!/usr/bin/env bash
# FUSION-LLaMA3.1-8B on-start script for Vast.ai PyTorch template.
# Paste the entire contents of this file into the "On-start Script" field.
set -euo pipefail

# ============================================================================
# Stage 1 — environment setup
# ============================================================================
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.hf_cache

SYS_PY=/usr/bin/python3
WEBUI_VENV=/workspace/.venvs/webui
LOGS=/workspace/logs

mkdir -p "$LOGS" "$HF_HOME"

apt-get update -qq
apt-get install -y -qq git tmux curl ca-certificates build-essential

echo "[stage 1] env ready | sys-py: $($SYS_PY --version)"

# ============================================================================
# Stage 2 — FUSION + FastChat install (system Python)
# Idempotent: skips if `fusion` is already importable.
# ============================================================================
if ! "$SYS_PY" -c "import fusion" 2>/dev/null; then
  echo "[stage 2] installing FUSION + FastChat …"
  cd /workspace
  if [ ! -d /workspace/FUSION ]; then
    git clone --depth 1 https://github.com/starriver030515/FUSION.git
  fi
  cd /workspace/FUSION
  "$SYS_PY" -m pip install --quiet -e .

  # fschat pulls an old transformers; use --no-deps and hand-pick runtime deps.
  "$SYS_PY" -m pip install --quiet --no-deps "fschat==0.2.36"
  "$SYS_PY" -m pip install --quiet shortuuid uvicorn fastapi httpx \
      "markdown2[all]" pydantic sse-starlette prompt_toolkit rich

  echo "[stage 2] FUSION + FastChat installed"
else
  echo "[stage 2] skipped — FUSION already installed"
fi

# ============================================================================
# Stage 3 — Open WebUI venv (isolated from system Python)
# Idempotent: skips if the `open-webui` binary already exists.
# ============================================================================
if [ ! -x "$WEBUI_VENV/bin/open-webui" ]; then
  echo "[stage 3] creating Open WebUI venv …"
  mkdir -p "$(dirname "$WEBUI_VENV")"
  "$SYS_PY" -m venv "$WEBUI_VENV"
  "$WEBUI_VENV/bin/pip" install --quiet --upgrade pip
  "$WEBUI_VENV/bin/pip" install --quiet open-webui
  echo "[stage 3] Open WebUI venv ready"
else
  echo "[stage 3] skipped — Open WebUI venv already exists"
fi

# ============================================================================
# Stage 4 — controller (21001) + FUSION model_worker (21002)
# ============================================================================
tmux kill-session -t controller 2>/dev/null || true
tmux new -d -s controller "\
  '$SYS_PY' -m fastchat.serve.controller \
    --host 127.0.0.1 --port 21001 \
  2>&1 | tee '$LOGS/controller.log'"

# Wait until the controller is actually accepting HTTP requests.
until curl -sf http://127.0.0.1:21001/list_models >/dev/null 2>&1; do sleep 1; done

tmux kill-session -t worker 2>/dev/null || true
tmux new -d -s worker "\
  ATTN_IMPLEMENTATION=sdpa \
  HF_HOME='$HF_HOME' \
  '$SYS_PY' -m fusion.serve.model_worker \
    --host 127.0.0.1 --port 21002 \
    --controller http://127.0.0.1:21001 \
    --worker http://127.0.0.1:21002 \
    --model-path starriver030515/FUSION-LLaMA3.1-8B \
    --model-name fusion-llama3-8b \
  2>&1 | tee '$LOGS/worker.log'"

echo "[stage 4] controller + worker launched"

# ============================================================================
# Stage 5 — FastChat OpenAI-compatible API server (8000)
# Waits until the worker has registered the model, not just until the
# controller is up.
# ============================================================================
tmux kill-session -t api 2>/dev/null || true
tmux new -d -s api "\
  until curl -sf http://127.0.0.1:21001/list_models | grep -q fusion-llama3-8b; do \
    echo 'waiting for worker to register …'; sleep 5; \
  done; \
  '$SYS_PY' -m fastchat.serve.openai_api_server \
    --controller-address http://127.0.0.1:21001 \
    --host 127.0.0.1 --port 8000 \
  2>&1 | tee '$LOGS/api.log'"

echo "[stage 5] OpenAI API adapter scheduled"

# ============================================================================
# Stage 6 — Open WebUI (3000)
# Waits for the OpenAI adapter to expose the model, then launches.
# ============================================================================
tmux kill-session -t webui 2>/dev/null || true
tmux new -d -s webui "\
  until curl -sf http://127.0.0.1:8000/v1/models | grep -q fusion-llama3-8b; do \
    echo 'waiting for API adapter to list model …'; sleep 5; \
  done; \
  ENABLE_BASE_MODELS_CACHE=false \
  OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEY=sk-dummy \
  WEBUI_AUTH=True \
  '$WEBUI_VENV/bin/open-webui' serve --host 127.0.0.1 --port 3000 \
  2>&1 | tee '$LOGS/webui.log'"

echo "[stage 6] Open WebUI scheduled"
echo "[done] startup sequence kicked off — check 'tmux ls' and tail logs in $LOGS/"
