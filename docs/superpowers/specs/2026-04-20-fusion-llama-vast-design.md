# FUSION-LLaMA3.1-8B on Vast.ai — Template Design

**Date:** 2026-04-20
**Status:** Approved (design phase)
**Related:** Prior spec for Qwen3-VL-8B Vast template (successfully running)

## 1. Goal

Provision a self-contained Vast.ai template that runs the multimodal model
[`starriver030515/FUSION-LLaMA3.1-8B`](https://huggingface.co/starriver030515/FUSION-LLaMA3.1-8B)
behind an Open WebUI frontend reachable from the user's laptop via SSH tunnel.

The template is **independent** of the existing Qwen3-VL-8B template —
clone on the Vast console, do not merge the two on-start scripts.

## 2. Constraints & Decisions

| Decision              | Choice                                                          |
|-----------------------|-----------------------------------------------------------------|
| Template strategy     | **Independent** (new Vast template, one model per instance)     |
| Frontend              | **Open WebUI** + FastChat OpenAI adapter (reuse upstream)       |
| GPU target            | **RTX 3090 / 4090** (24 GB VRAM)                                |
| Dtype                 | **BF16** (as documented on the FUSION HF card)                  |
| Persistence           | **None** (ephemeral, re-download weights on restart)            |
| Automation            | Hybrid — inline on-start script, idempotent                     |
| Exposure              | **SSH tunnel only** (no public ports except `:22`)              |

## 3. Key Research Findings

- FUSION has a **custom architecture** (SigLIP vision encoder + Llama-3.1-8B LLM
  with text-guided fusion). It is **not supported by vLLM**.
- Upstream ships a FastChat-style serving stack: `controller` +
  `model_worker` + `gradio_web_server`. **No OpenAI-compatible server.**
- FUSION's `model_worker` implements the **FastChat worker protocol**
  (`/worker_generate_stream`, `/worker_get_status`, base64 images in JSON).
- Because of that, `fastchat.serve.openai_api_server` from the upstream `fschat`
  package can proxy OpenAI `/v1/chat/completions` → controller → worker
  **without any custom adapter code**.
- FUSION pins `transformers==4.48.1`, `accelerate==1.3.0`. These conflict with
  Open WebUI's `transformers>=5.x` → the two must live in separate Python
  environments.
- FUSION's README suggests `pip install flash-attn --no-build-isolation`, but
  flash-attn's pre-built wheels carry a CUDA PTX version mismatch on many
  Vast hosts (same class of failure seen with the Qwen3-VL template). We skip
  the flash-attn install and force SDPA.

## 4. Architecture

```
                    ┌──────────────────────── user's laptop
                    │  browser → http://localhost:3000
                    │                    │
                    │  ssh -L 3000:127.0.0.1:3000 user@vast …
                    └────────────────────┼────────── Vast instance
                                         ▼
 ┌─────────────┐   OpenAI  ┌──────────┐  RPC   ┌────────────┐ heartbeat ┌──────────┐
 │ open-webui  │ ───────►  │ fastchat │ ─────► │ controller │ ◄───────► │ worker   │
 │  :3000      │   /v1/*   │  api     │        │  :21001    │           │  :21002  │
 │ (venv A)    │           │  :8000   │        │ (sys-py)   │           │ FUSION   │
 │             │           │ (sys-py) │        │            │           │ (sys-py) │
 └─────────────┘           └──────────┘        └────────────┘           └──────────┘
                                                                        GPU: ~20GB BF16
```

### 4.1 Components

| Component   | Process                                   | Port  | Python env              |
|-------------|-------------------------------------------|-------|-------------------------|
| controller  | `fastchat.serve.controller`               | 21001 | system (`/usr/bin/python3`) |
| worker      | `fusion.serve.model_worker`               | 21002 | system                  |
| api         | `fastchat.serve.openai_api_server`        | 8000  | system                  |
| webui       | `open-webui serve`                        | 3000  | `/workspace/.venvs/webui` |

Each runs in its own **tmux** session (`tmux new -d -s <name>`) so the user
can `tmux attach` for live logs. Logs also tee to `/workspace/logs/*.log`.

### 4.2 Isolation

Two Python environments on the instance:

- **System Python** (`/usr/bin/python3`): holds FUSION, FastChat, and all the
  ML deps (transformers 4.48.1, torch, siglip, timm, …).
- **Open WebUI venv** (`/workspace/.venvs/webui`): created with `python3 -m venv`,
  holds only `open-webui` and its pinned transformers ≥ 5.x.

The two only communicate over HTTP (`http://127.0.0.1:8000/v1/…`), never
over Python imports — so their dependency trees cannot collide.

## 5. On-start Script Stages

The script is **idempotent** (checks before install/clone) and runs top-to-bottom
on every container start.

### Stage 1 — Env setup
```bash
set -e
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/logs /workspace/.hf_cache
export HF_HOME=/workspace/.hf_cache
SYS_PY=/usr/bin/python3
WEBUI_VENV=/workspace/.venvs/webui
apt-get update -qq && apt-get install -y -qq git tmux curl ca-certificates
```

### Stage 2 — FUSION + FastChat install (system Python)
```bash
if ! $SYS_PY -c "import fusion" 2>/dev/null; then
  cd /workspace && git clone https://github.com/starriver030515/FUSION.git
  cd FUSION && $SYS_PY -m pip install -e . --quiet
  $SYS_PY -m pip install "fschat[model_worker,webui]==0.2.36" --quiet --no-deps
  $SYS_PY -m pip install shortuuid uvicorn fastapi httpx markdown2 --quiet
fi
```
Notes:
- `--no-deps` on `fschat` prevents pinning an older `transformers` that would
  fight FUSION's `4.48.1`.
- Missing transitive deps (`shortuuid`, `uvicorn`, `fastapi`, `httpx`, `markdown2`)
  are installed explicitly afterwards.
- **No** `pip install flash-attn` (PTX risk).

### Stage 3 — Open WebUI venv
```bash
if [ ! -x "$WEBUI_VENV/bin/open-webui" ]; then
  $SYS_PY -m venv $WEBUI_VENV
  $WEBUI_VENV/bin/pip install -q --upgrade pip
  $WEBUI_VENV/bin/pip install -q open-webui
fi
```

### Stage 4 — Controller + worker
```bash
tmux new -d -s controller "$SYS_PY -m fastchat.serve.controller \
  --host 127.0.0.1 --port 21001 2>&1 | tee /workspace/logs/controller.log"

sleep 3

tmux new -d -s worker "\
  ATTN_IMPLEMENTATION=sdpa \
  $SYS_PY -m fusion.serve.model_worker \
    --host 127.0.0.1 --port 21002 \
    --controller http://127.0.0.1:21001 \
    --worker http://127.0.0.1:21002 \
    --model-path starriver030515/FUSION-LLaMA3.1-8B \
    --model-name fusion-llama3-8b \
  2>&1 | tee /workspace/logs/worker.log"
```

### Stage 5 — OpenAI-compatible API
```bash
tmux new -d -s api "\
  until curl -sf http://127.0.0.1:21001/list_models >/dev/null; do sleep 5; done; \
  $SYS_PY -m fastchat.serve.openai_api_server \
    --controller-address http://127.0.0.1:21001 \
    --host 127.0.0.1 --port 8000 \
  2>&1 | tee /workspace/logs/api.log"
```

### Stage 6 — Open WebUI
```bash
tmux new -d -s webui "\
  until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done; \
  ENABLE_BASE_MODELS_CACHE=false \
  OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
  OPENAI_API_KEY=sk-dummy \
  WEBUI_AUTH=True \
  $WEBUI_VENV/bin/open-webui serve --host 127.0.0.1 --port 3000 \
  2>&1 | tee /workspace/logs/webui.log"
```

## 6. Ports & SSH Tunnel

### 6.1 Docker Options (Vast template field)
```
-p 22:22
```
Only SSH is exposed publicly. Every service binds `127.0.0.1`.

### 6.2 Internal port map

| Port  | Service                   | Bind      | How reached from user |
|-------|---------------------------|-----------|-----------------------|
| 21001 | FastChat controller       | 127.0.0.1 | internal only         |
| 21002 | FUSION model_worker       | 127.0.0.1 | internal only         |
| 8000  | FastChat OpenAI adapter   | 127.0.0.1 | optional SSH `-L`     |
| 3000  | Open WebUI                | 127.0.0.1 | **SSH `-L 3000:127.0.0.1:3000`** |
| 22    | SSH                       | 0.0.0.0   | Vast-assigned port    |

### 6.3 SSH tunnel

```bash
ssh -p <VAST_SSH_PORT> -L 3000:127.0.0.1:3000 root@<VAST_HOST>
```

Optional debug tunnel (also forward raw OpenAI API):
```bash
ssh -p <VAST_SSH_PORT> -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 root@<VAST_HOST>
```

Then on the laptop: open `http://localhost:3000`.

## 7. Risks & Mitigations

| # | Risk                                                | Mitigation                                                                           |
|---|-----------------------------------------------------|--------------------------------------------------------------------------------------|
| 1 | flash-attn PTX mismatch (prebuilt wheel too new)    | Skip `pip install flash-attn`; set `ATTN_IMPLEMENTATION=sdpa` when launching worker. |
| 2 | `transformers` version conflict with Open WebUI     | Two separate Python envs; HTTP-only interface.                                       |
| 3 | FastChat's OpenAI adapter may not forward images    | Smoke-test text first; if image path fails, add a ~30-LOC FastAPI shim at `/workspace/patches/vision_forward.py`. |
| 4 | `fschat` pulls conflicting transformers             | `pip install fschat --no-deps`, then install transitive deps explicitly.             |
| 5 | Worker registers before controller is up            | `sleep 3` between stages; `until curl -sf` guards on the API and WebUI starts.       |
| 6 | Container rebuild wipes HF cache → re-download 20GB | Accepted (user chose no persistence); filter Vast for `Inet Down ≥ 500 Mbps`.        |

**Fallback if Open WebUI image pipeline is broken:** launch
`fusion.serve.gradio_web_server --controller http://127.0.0.1:21001 --port 7860`
as a 5th tmux session and tunnel `-L 7860:127.0.0.1:7860`. Text chat stays on
Open WebUI, image chat via Gradio.

## 8. Step-by-step: modifying the Vast template

1. Go to https://cloud.vast.ai/templates/.
2. Find **"PyTorch (Vast)"** (or `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`)
   → **Edit Template** → **Save as new**.
3. Name it `FUSION-LLaMA3.1-8B`.
4. Set fields:
   - **Image Path:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
   - **Docker Options:** `-p 22:22`
   - **Launch Mode:** `ssh`
   - **On-start Script:** paste the script produced during implementation.
   - **Environment Variables:** leave blank (script sets its own).
   - **Disk Space:** `50 GB`.
5. Save.
6. In the instance-search page, filter:
   - GPU: `RTX 3090` or `RTX 4090` (24 GB)
   - CUDA Version: `≥ 12.4`
   - Disk Space: `≥ 50 GB`
   - Inet Down: `≥ 500 Mbps`
7. Rent an instance, wait for "Running" + green tick.
8. Copy the SSH command from the **Connect** dialog; edit it to add
   `-L 3000:127.0.0.1:3000`.
9. Monitor: `tmux ls`, `tail -f /workspace/logs/worker.log` until
   `Uvicorn running on … 21002`, then `/workspace/logs/webui.log` until
   `Application startup complete`. Total ~10-15 min on first boot.
10. Browser → `http://localhost:3000` → sign up admin → pick `fusion-llama3-8b` → chat.

## 9. Success criteria

- `tmux ls` on the running instance shows all four sessions alive.
- `curl http://127.0.0.1:21001/list_models` on the instance returns
  `{"models": ["fusion-llama3-8b"]}`.
- `curl http://127.0.0.1:8000/v1/models` on the instance returns an
  OpenAI-shaped list including `fusion-llama3-8b`.
- From the laptop browser, `http://localhost:3000` loads Open WebUI and a
  text chat with `fusion-llama3-8b` returns a coherent response.
- Image upload smoke test: if it succeeds, image pipeline works end-to-end;
  if it fails, we apply the vision-shim mitigation from §7.

## 10. Out of scope

- Multi-model switching on one instance (deliberately rejected — Template C choice).
- Quantization (INT8/FP8) — user chose BF16.
- Persistent weight storage (Vast volume) — user chose ephemeral.
- Public exposure of any service other than SSH.
- Fine-tuning or LoRA workflow.
- Automatic Vast instance creation via API (the user drives the UI by hand).
