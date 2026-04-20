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

sleep 3

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
