# Open WebUI - Qwen3-VL 8B (vastai/vllm) — Vast Template

A Vast.ai template that serves Qwen3-VL-8B-Instruct via vLLM and exposes
Open WebUI for chat. Built on top of `vastai/vllm:v0.20.0-cuda-13.0`, so
vLLM ships pre-installed and starts via the image's supervisord. The
`onstart.sh` in this directory only installs and launches Open WebUI plus
the persistent secret/data layout.

## Files

| File | Purpose |
|---|---|
| `onstart.sh` | Pasted into the Vast template's "On-start Script" field |
| `onstart.sh.bak-pre-vllm-migration` | Snapshot of the previous (vastai/pytorch-based) onstart, kept for archaeology |
| `TEMPLATE_FIELDS.md` | Copy-paste source for every Vast Console field |
| `functions/qwenvl_image_compress.py` | Open WebUI Filter Function (paste into Admin → Functions after first boot) |
| `functions/test_*.py` | pytest suite (66 tests) for the filter, runs locally |
| `scripts/dump_captions.py` | CLI to inspect the filter's caption cache on the pod |
| `scripts/integration_test_image_compressor.sh` | Smoke test for the filter end-to-end on the pod |

## Deploy procedure

### 1. Configure the Vast template (one-time)

1. Open Vast Console → **Templates → My Templates**.
2. Find the existing `PyTorch (Vast) + Open WebUI - Qwen3-VL 8B (v2)` template.
3. Take screenshots of every field (Identification, Image, Docker Options,
   Env Vars, On-start Script). Save them under `~/vast-template-backup/`
   on your laptop. This is your rollback artifact.
4. Click "..." → **Clone** (or "Save as new"). Name the clone with a
   `(vllm-base)` suffix.
5. Open the clone for editing.
6. Open `TEMPLATE_FIELDS.md` from this repo and paste each field
   exactly as shown.
7. Open `onstart.sh` from this repo and paste its entire contents into
   the "On-start Script" textarea.
8. **Save** (not Save & Use).

### 2. Rent a test instance

1. Templates → click "Use" on the new template.
2. Pick a GPU offer with ≥ 24 GB VRAM (RTX 3090, 4090, A5000, A6000).
3. Confirm the offer's storage is "persistent" / "Volume" so
   `/workspace` survives stop/start.
4. Rent.
5. SSH with port forwards as soon as the pod shows `running`:
   ```bash
   ssh -p <PORT> root@<HOST> -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000
   ```

### 3. Watch the boot

Open two SSH shells side by side:

```bash
# Shell 1: vLLM service (started by onstart, NOT supervisord, see note below)
tail -f /workspace/logs/vllm.log
```

```bash
# Shell 2: our onstart (full bring-up: secret, venv, vllm tmux, webui tmux)
tail -f /workspace/logs/onstart.log
# (Vast also mirrors the same content to /var/log/onstart.log)
```

Wait until both show their respective "ready" lines:
- vLLM: `Application startup complete` (or `Uvicorn running on http://127.0.0.1:8000`)
- Onstart: `=== Bootstrap done. SSH tunnel: ssh -L 3000:127.0.0.1:3000 ... ===`

Cold boot expected: 10-20 minutes (mostly Qwen3-VL weights download
~16 GB into `/workspace/.hf_cache`).
Warm boot (subsequent restart): ~2 minutes (everything persists).

> **Why `onstart.sh` launches vLLM directly (not via supervisord):** when this
> template is created by cloning a vastai/pytorch-based template (the typical
> path), Vast preserves the legacy `/.launch` boot mode, which **does not** run
> the image's `boot_default.sh` chain. As a result the image's supervisord and
> its `vllm.sh`/`ray.sh`/`caddy` services never start. Instead `onstart.sh`
> starts vLLM itself under a `tmux` session named `vllm`. The script's
> Step `[2/3]` short-circuits if a vLLM is already serving on `:8000` (so the
> same script also works against the modern entrypoint mode if you ever
> migrate the template). See `/workspace/logs/vllm.log` for the live log.

### 4. Verify Open WebUI

In your laptop browser open `http://localhost:3000` (via the SSH tunnel).
- First visit: register the admin account.
- **Admin → Models** should auto-list `qwen3-vl-8b` (discovered from the
  image-managed vLLM at `127.0.0.1:8000`).

### 5. Install the image-aware compressor filter

1. **Admin → Functions → +** (create new function).
2. On your laptop, open `vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py`
   from this repo. Copy the entire file contents.
3. Paste into the Vast UI's function editor.
4. Name: `Qwen3-VL Image Compressor`. Save.
5. **Admin → Models → qwen3-vl-8b → Filters tab** → tick the new filter →
   Save.

### 6. Smoke test the filter

In a chat with model `qwen3-vl-8b`:
1. Upload one image, ask a question, send.
2. Confirm the response includes a `<details>🧠 Image compressor reasoning…</details>`
   block (collapsed by default; click to expand).
3. SSH to the pod:
   ```bash
   sqlite3 /workspace/openwebui-data/img_captions.db "SELECT count(*) FROM captions;"
   ```
   Expected: ≥ 1.
4. Send a text-only follow-up that changes topic. Confirm the next
   response's reasoning block shows `🎯 Router: drop images, saved ~XXXX tokens`.

If any of the above fails: see "Troubleshooting" below.

### 7. Rollback

- **You cloned the template (recommended)**: just don't use the new
  template. The original `(v2)` template is untouched. To clean up later,
  delete the clone.
- **You edited in place**: restore from the screenshots taken in step 1.
  This is mechanical but tedious — argues for the clone-first path.

## Operations

### Restart only Open WebUI (don't touch vLLM)

```bash
ssh -p <PORT> root@<HOST>
tmux kill-session -t webui
bash /var/lib/vast/onstart.sh   # path may differ; the file is the on-start script you pasted
```

The script is idempotent: it skips the venv install (already present),
skips the secret-key generation (already present), waits 0 seconds for
vLLM (already healthy), restarts the `webui` tmux session.

### Restart only vLLM

```bash
ssh -p <PORT> root@<HOST>
tmux kill-session -t vllm
bash /var/lib/vast/onstart.sh
```

Onstart's idempotent vLLM block sees that nothing is on `:8000` and re-launches
the `vllm` tmux session. The Open WebUI `webui` tmux session is also recreated
by the same script (which is harmless — it just restarts the existing webui).

If you migrate to the modern entrypoint mode in a future template (where the
image's supervisord owns vLLM), use `supervisorctl restart vllm` instead.

### Inspect filter activity

```bash
# Per-inlet JSON-line log (in webui.log)
grep image_compress_inlet /workspace/logs/webui.log | tail -10 | jq

# Caption cache contents
python /workspace/scripts/dump_captions.py /workspace/openwebui-data/img_captions.db
# (or copy the script in if not present)
```

### Stop Open WebUI but keep vLLM (e.g. for debugging)

Edit the template's `OPENWEBUI_ENABLE` env var to `false`, then restart
the instance. vLLM still runs. The onstart sees the flag and exits early
without touching Open WebUI.

## Troubleshooting

| Symptom | Check | Likely cause |
|---|---|---|
| Onstart hangs at `Waiting for vLLM /health` for >20 min | `tmux ls` + `tail -100 /workspace/logs/vllm.log` | vLLM tmux either crashed (bad flag, OOM) or is still downloading model weights (16 GB cold) |
| `vllm.log` shows `unrecognized argument: --mm-encoder-attn-backend` | n/a | vLLM 0.20 dropped the flag — remove it from `VLLM_ARGS` env var **and** from `onstart.sh`'s hardcoded args block |
| `vllm.log` shows `Engine core initialization failed` after second `APIServer pid=...` line | `ss -tlnp \| grep 8000` | A second vllm instance tried to bind `:8000` (e.g. you re-added `entrypoint.sh` to the end of onstart). Kill the duplicate; only one vllm should own port 8000 |
| Open WebUI loads but no model in Admin → Models | `curl -sf http://127.0.0.1:8000/v1/models` | vLLM API not reachable; check the `vllm` tmux is alive (`tmux ls`) |
| Filter not running on chat | `grep image_compress_inlet /workspace/logs/webui.log \| tail` | Filter not enabled on the model, or function disabled |
| First chat 503s | `tail -f /workspace/logs/webui.log` | vLLM weights still loading (cold boot); wait 1-2 min |
| Captions cache empty after several chats | `sqlite3 ... "SELECT count(*) FROM captions;"` | Filter may be disabled per-user (`UserValves.enabled=false`) or per-model (Models → qwen3-vl-8b → Filters unticked) |
| Sessions die on restart | `cat /workspace/.webui-secret` | File missing → secret regenerated each boot → JWTs invalidated. Rerun onstart, the file should exist. |
| `/workspace` lost after stop/start | Vast offer details | Storage type was ephemeral, not persistent. Use a different offer. |
| Open WebUI fails to bind `:3000` with `address already in use` | `ss -tlnp \| grep 3000` | The image's `caddy` proxy (started when `entrypoint.sh` runs) is on `:3000`. Either remove `entrypoint.sh` from onstart, or change Open WebUI's `--port` to `13000` and let caddy reverse-proxy from `:3000` |
