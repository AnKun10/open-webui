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
