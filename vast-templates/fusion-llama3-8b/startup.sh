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
