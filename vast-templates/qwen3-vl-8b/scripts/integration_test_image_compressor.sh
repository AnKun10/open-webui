#!/usr/bin/env bash
# Run on a Vast pod that has Open WebUI + vLLM up and the filter installed.
# Requires: API token from Open WebUI Settings → Account → API Keys.
set -eu

BASE="${OPENWEBUI_BASE:-http://127.0.0.1:3000}"
TOKEN="${OPENWEBUI_TOKEN:?Set OPENWEBUI_TOKEN to a valid API key}"
DB="${CAPTION_DB:-/workspace/openwebui-data/img_captions.db}"

count_captions() {
	sqlite3 "$DB" "SELECT count(*) FROM captions;" 2>/dev/null || echo 0
}

before=$(count_captions)
echo "Captions in cache before: $before"

# Scenario 1: upload 1 small synthetic PNG + ask
PNG_B64=$(printf '\x89PNG\r\n\x1a\nfake' | base64 | tr -d '\n')
read -r -d '' PAYLOAD <<EOF || true
{
  "model": "qwen3-vl-8b",
  "stream": false,
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "this is a smoke test image"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${PNG_B64}"}}
    ]}
  ]
}
EOF
curl -fsS -X POST "$BASE/api/chat/completions" \
	-H "Authorization: Bearer $TOKEN" \
	-H "Content-Type: application/json" \
	-d "$PAYLOAD" >/dev/null
sleep 2

after=$(count_captions)
echo "Captions in cache after: $after"
if [ "$after" -gt "$before" ]; then
	echo "PASS: caption row added"
else
	echo "FAIL: no new caption row"
	exit 1
fi
