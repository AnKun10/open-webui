# FUSION-LLaMA3.1-8B Vast.ai Template

On-start script and docs for running [`starriver030515/FUSION-LLaMA3.1-8B`](https://huggingface.co/starriver030515/FUSION-LLaMA3.1-8B) on a Vast.ai PyTorch instance, reachable from your laptop via SSH tunnel to Open WebUI.

See the full design in `docs/superpowers/specs/2026-04-20-fusion-llama-vast-design.md`.

## What boots on the instance

| Port  | Service                     | Bind      | Public? |
|-------|-----------------------------|-----------|---------|
| 21001 | FastChat controller         | 127.0.0.1 | no      |
| 21002 | FUSION model_worker         | 127.0.0.1 | no      |
| 8000  | FastChat OpenAI adapter     | 127.0.0.1 | no      |
| 3000  | Open WebUI                  | 127.0.0.1 | **via SSH tunnel** |
| 22    | SSH                         | 0.0.0.0   | Vast-assigned |

## Step-by-step — creating the template

1. Open https://cloud.vast.ai/templates/.
2. Find the base template **"PyTorch (Vast)"** (image `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) → **Edit Template** → **Save as new**.
3. Name it `FUSION-LLaMA3.1-8B`.
4. Fill fields:
   - **Image Path:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
   - **Docker Options:** `-p 22:22`
   - **Launch Mode:** `ssh`
   - **On-start Script:** paste the full contents of [`startup.sh`](./startup.sh) in this directory.
   - **Environment Variables:** leave empty.
   - **Disk Space:** `50 GB`.
5. Save.

## Step-by-step — renting an instance

1. On the instance search page, apply filters:
   - GPU: `RTX 3090` or `RTX 4090` (24 GB VRAM)
   - CUDA Version: `≥ 12.4`
   - Disk Space: `≥ 50 GB`
   - Inet Down: `≥ 500 Mbps`
2. Rent an instance, wait for status **Running** with green tick.
3. Click **Connect** → copy the SSH command.
4. Rewrite it to include a tunnel:
   ```bash
   ssh -p <VAST_SSH_PORT> -L 3000:127.0.0.1:3000 root@<VAST_HOST>
   ```

## Step-by-step — verifying the boot

Once you're on the instance over SSH:

```bash
# 1. Check all four tmux sessions exist
tmux ls
# expected: controller, worker, api, webui

# 2. Tail the slowest service (worker — it downloads ~20 GB of weights)
tail -f /workspace/logs/worker.log
# wait for: Uvicorn running on http://127.0.0.1:21002

# 3. When worker is ready, run the smoke test.
#    First, copy smoke-test.sh onto the instance from your laptop
#    (open a second local terminal):
#      scp -P <VAST_SSH_PORT> vast-templates/fusion-llama3-8b/smoke-test.sh \
#          root@<VAST_HOST>:/workspace/
#    Then, back on the instance:
bash /workspace/smoke-test.sh
# expected: ALL CHECKS PASSED
```

First-boot total time: **10–15 min** (model download dominates).

Subsequent restarts in the **same container**: ~1–2 min (install stages are skipped via idempotent guards; HF cache at `/workspace/.hf_cache` is re-used).

Container rebuilds wipe `/workspace`, so weights re-download on every new instance — expected; you chose "no persistence".

## Step-by-step — using the frontend

1. Keep the SSH tunnel open.
2. On your laptop, browse to `http://localhost:3000`.
3. Sign up an admin account on first boot.
4. In the model dropdown, pick `fusion-llama3-8b`.
5. Send a text message to confirm the pipeline is alive.
6. (Optional) Upload a small image. If Open WebUI can pass it through to the model, the vision pipeline works end-to-end. If the model responds with garbage or an error, apply the "vision shim" mitigation from the spec (see [§7 Risks & mitigations](../../docs/superpowers/specs/2026-04-20-fusion-llama-vast-design.md#7-risks--mitigations)).

## Debugging cheatsheet

```bash
tmux attach -t worker          # live logs, Ctrl+b d to detach
tmux attach -t api
tmux attach -t webui

curl http://127.0.0.1:21001/list_models                         # controller view
curl http://127.0.0.1:8000/v1/models                            # API view
curl -X POST http://127.0.0.1:8000/v1/chat/completions \        # raw text test
     -H "Content-Type: application/json" \
     -d '{"model":"fusion-llama3-8b","messages":[{"role":"user","content":"hello"}]}'

nvidia-smi                     # GPU utilisation / VRAM
```

## Known risks and mitigations

See the spec §7 for the full table. Summary:

- **flash-attn PTX mismatch** → we skip its install; `ATTN_IMPLEMENTATION=sdpa` forces the SDPA fallback.
- **transformers version clash with Open WebUI** → separate Python envs.
- **FastChat OpenAI adapter may not forward images** → smoke-test text first; if images fail, add a ~30-LOC shim at `/workspace/patches/vision_forward.py` or fall back to FUSION's own Gradio UI on port 7860.
