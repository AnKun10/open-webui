#!/usr/bin/env bash
# Run on the Vast instance after boot to verify the stack is alive.
set -uo pipefail

FAIL=0

check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  \u2713 %s\n' "$label"
  else
    printf '  \u2717 %s  (cmd: %s)\n' "$label" "$*"
    FAIL=1
  fi
}

echo "=== tmux sessions ==="
tmux ls || true
for s in controller worker api webui; do
  check "tmux session: $s" tmux has-session -t "$s"
done

echo
echo "=== internal HTTP checks ==="
check "controller /list_models HTTP 200"  curl -sf http://127.0.0.1:21001/list_models
check "controller lists fusion-llama3-8b" bash -c 'curl -sf http://127.0.0.1:21001/list_models | grep -q fusion-llama3-8b'
check "API /v1/models HTTP 200"           curl -sf http://127.0.0.1:8000/v1/models
check "API lists fusion-llama3-8b"        bash -c 'curl -sf http://127.0.0.1:8000/v1/models | grep -q fusion-llama3-8b'
check "WebUI /health HTTP 200"            curl -sf http://127.0.0.1:3000/health

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || FAIL=1

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "SOME CHECKS FAILED — inspect /workspace/logs/*.log"
  echo "  tail -n 50 /workspace/logs/worker.log"
  echo "  tail -n 50 /workspace/logs/api.log"
  echo "  tail -n 50 /workspace/logs/webui.log"
  exit 1
fi
