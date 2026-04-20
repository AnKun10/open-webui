# FUSION-LLaMA3.1-8B Vast.ai Template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a committed, committable `startup.sh` + README pair that the user can paste into a Vast.ai "PyTorch (Vast)" template to run FUSION-LLaMA3.1-8B behind Open WebUI, reachable from the user's laptop over SSH tunnel.

**Architecture:** A single on-start shell script launches 4 tmux-backed services (FastChat controller, FUSION model_worker, FastChat OpenAI API adapter, Open WebUI) with two isolated Python environments. No vLLM — FUSION's custom architecture is incompatible. FastChat's upstream `openai_api_server` is the OpenAI adapter (no custom code needed).

**Tech Stack:** Bash, tmux, Python 3 (system + isolated venv), FastChat, Open WebUI, CUDA/PyTorch 2.4 + cuda12.4 base image on Vast.ai.

**Testing approach:** This is a shell artifact for a remote GPU environment — we cannot unit-test it locally. The plan substitutes:
1. Static validation (`bash -n`, `shellcheck`) run locally.
2. A post-boot smoke-test script (`smoke-test.sh`) the user runs on the instance.
3. An explicit manual end-to-end checklist.

---

## Files

- Create: `vast-templates/fusion-llama3-8b/startup.sh` — the on-start script (pasted verbatim into Vast template).
- Create: `vast-templates/fusion-llama3-8b/smoke-test.sh` — post-boot validation the user runs on the instance.
- Create: `vast-templates/fusion-llama3-8b/README.md` — step-by-step modify-template guide, mirrors spec §8.
- Create: `vast-templates/fusion-llama3-8b/.gitignore` — keep local experiments out of git.

Each file has one clear responsibility:
- `startup.sh` = orchestrate install + launch.
- `smoke-test.sh` = verify the 4 services are alive and the model is listed end-to-end.
- `README.md` = human-facing operating manual.

---

## Task 1: Scaffolding

**Files:**
- Create: `vast-templates/fusion-llama3-8b/startup.sh`
- Create: `vast-templates/fusion-llama3-8b/smoke-test.sh`
- Create: `vast-templates/fusion-llama3-8b/README.md`
- Create: `vast-templates/fusion-llama3-8b/.gitignore`

- [ ] **Step 1: Create the directory**

Run:
```bash
mkdir -p vast-templates/fusion-llama3-8b
```

- [ ] **Step 2: Create placeholder files with shebang/title lines only**

Write to `vast-templates/fusion-llama3-8b/startup.sh`:
```bash
#!/usr/bin/env bash
# FUSION-LLaMA3.1-8B on-start script for Vast.ai PyTorch template.
# Paste the entire contents of this file into the "On-start Script" field.
set -euo pipefail
```

Write to `vast-templates/fusion-llama3-8b/smoke-test.sh`:
```bash
#!/usr/bin/env bash
# Run on the Vast instance after boot to verify the stack is alive.
set -euo pipefail
```

Write to `vast-templates/fusion-llama3-8b/README.md`:
```markdown
# FUSION-LLaMA3.1-8B Vast.ai Template

See `docs/superpowers/specs/2026-04-20-fusion-llama-vast-design.md` for the design.
```

Write to `vast-templates/fusion-llama3-8b/.gitignore`:
```
*.local.sh
*.bak
```

- [ ] **Step 3: Make shell scripts executable**

Run:
```bash
chmod +x vast-templates/fusion-llama3-8b/startup.sh vast-templates/fusion-llama3-8b/smoke-test.sh
```

- [ ] **Step 4: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/
git commit -m "scaffold: FUSION-LLaMA3.1-8B Vast template directory"
```

---

## Task 2: Stage 1 — environment setup

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 1 block)

- [ ] **Step 1: Append Stage 1 to `startup.sh`**

Add the following after the existing `set -euo pipefail` line (keep everything already there):

```bash

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
```

- [ ] **Step 2: Static-validate the script**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected output: (none — zero exit code means syntax OK).

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 1 — env setup (dirs, apt packages, env vars)"
```

---

## Task 3: Stage 2 — FUSION + FastChat install

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 2 block)

Why `--no-deps` on fschat: the `fschat==0.2.36` wheel pins an older `transformers` that conflicts with FUSION's `transformers==4.48.1`. We install fschat without its dep tree and then pull in the small set of transitive libs fschat's worker/server code actually imports.

Why no flash-attn: the prebuilt flash-attn wheel ships CUDA PTX that is often newer than the driver on Vast hosts (same failure class the Qwen template hit). FUSION falls back to SDPA when flash-attn is absent, triggered via `ATTN_IMPLEMENTATION=sdpa`.

- [ ] **Step 1: Append Stage 2 to `startup.sh`**

```bash

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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected: zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 2 — install FUSION repo and FastChat server"
```

---

## Task 4: Stage 3 — Open WebUI venv

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 3 block)

- [ ] **Step 1: Append Stage 3 to `startup.sh`**

```bash

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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected: zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 3 — isolated Open WebUI venv"
```

---

## Task 5: Stage 4 — controller + worker tmux launches

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 4 block)

`ATTN_IMPLEMENTATION=sdpa` forces PyTorch SDPA (no flash-attn dependency). After launching the controller, an `until curl -sf .../list_models` poll lets its HTTP server bind before the worker tries to register.

- [ ] **Step 1: Append Stage 4 to `startup.sh`**

```bash

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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected: zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 4 — launch controller and FUSION worker in tmux"
```

---

## Task 6: Stage 5 — OpenAI-compatible API

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 5 block)

The guard greps for `fusion-llama3-8b` specifically, not just HTTP 200 — the controller's `/list_models` returns `{"models":[]}` with HTTP 200 while the worker is still downloading weights. The worker registers under its `--model-name` once weights are loaded, and only then does the grep succeed.

- [ ] **Step 1: Append Stage 5 to `startup.sh`**

```bash

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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected: zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 5 — FastChat OpenAI API adapter with worker-ready guard"
```

---

## Task 7: Stage 6 — Open WebUI

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (append Stage 6 block)

`WEBUI_AUTH=True` forces admin signup on first boot; `ENABLE_BASE_MODELS_CACHE=false` stops Open WebUI from caching the model list between restarts (we need fresh reads because the worker may not be ready when WebUI starts if boot ordering shifts).

- [ ] **Step 1: Append Stage 6 to `startup.sh`**

```bash

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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/startup.sh
```

Expected: zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/startup.sh
git commit -m "startup: stage 6 — Open WebUI with model-ready guard"
```

---

## Task 8: Shellcheck the complete script

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/startup.sh` (fix anything shellcheck flags)

- [ ] **Step 1: Install shellcheck if missing**

Run:
```bash
shellcheck --version || (echo "Install shellcheck from https://www.shellcheck.net/ first, then re-run." && exit 1)
```

Expected: version line printed.

- [ ] **Step 2: Run shellcheck**

Run:
```bash
shellcheck vast-templates/fusion-llama3-8b/startup.sh
```

- [ ] **Step 3: Fix any warnings**

Common expected warnings and how to resolve:
- **SC2086** (unquoted expansion): wrap the variable in double quotes — e.g. `"$LOGS"`.
- **SC2046** (word splitting on command substitution): same fix — quote it.
- **SC2016** (expressions in single quotes): the tmux command strings are intentionally embedding `$VAR` into a literal that tmux later evaluates; suppress with an inline `# shellcheck disable=SC2016` comment directly above the affected `tmux new -d -s …` line if needed.

Edit `startup.sh` in place for each warning. Re-run `shellcheck` until it's clean.

- [ ] **Step 4: Commit**

Only commit if anything changed. Run:
```bash
git diff --quiet vast-templates/fusion-llama3-8b/startup.sh || {
  git add vast-templates/fusion-llama3-8b/startup.sh
  git commit -m "startup: address shellcheck warnings"
}
```

---

## Task 9: Smoke-test script

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/smoke-test.sh` (replace placeholder with full body)

This script runs **on the Vast instance after boot** and returns non-zero on any failed check. The user can tail the output and re-run until everything is green.

- [ ] **Step 1: Replace `smoke-test.sh` body**

Write to `vast-templates/fusion-llama3-8b/smoke-test.sh`:

```bash
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
```

- [ ] **Step 2: Static-validate**

Run:
```bash
bash -n vast-templates/fusion-llama3-8b/smoke-test.sh
shellcheck vast-templates/fusion-llama3-8b/smoke-test.sh
```

Expected: both zero exit code.

- [ ] **Step 3: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/smoke-test.sh
git commit -m "smoke-test: post-boot validation of tmux sessions and HTTP endpoints"
```

---

## Task 10: README — step-by-step modify-template guide

**Files:**
- Modify: `vast-templates/fusion-llama3-8b/README.md` (replace placeholder with full body)

This mirrors §8 of the spec and is the user-facing operating manual.

- [ ] **Step 1: Replace `README.md` body**

Write to `vast-templates/fusion-llama3-8b/README.md`:

````markdown
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

# 3. When worker is ready, run the smoke test
bash /workspace/path/to/smoke-test.sh   # copy smoke-test.sh onto the instance
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
6. (Optional) Upload a small image. If Open WebUI can pass it through to the model, the vision pipeline works end-to-end. If the model responds with garbage or an error, apply the "vision shim" mitigation from the spec (see Risks §7).

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
````

- [ ] **Step 2: Commit**

Run:
```bash
git add vast-templates/fusion-llama3-8b/README.md
git commit -m "docs: README for FUSION-LLaMA3.1-8B Vast template"
```

---

## Task 11: Manual end-to-end verification

This task is manual — not executed by an agent. The user performs it on Vast.ai and reports back.

- [ ] **Step 1: Save-as-new the PyTorch (Vast) template**

Follow the README steps under "creating the template". Paste the committed `startup.sh` verbatim into the On-start Script field.

- [ ] **Step 2: Rent an RTX 3090/4090 instance**

Apply the filters from the README. Launch.

- [ ] **Step 3: SSH in with the tunnel**

```bash
ssh -p <VAST_SSH_PORT> -L 3000:127.0.0.1:3000 root@<VAST_HOST>
```

- [ ] **Step 4: Copy the smoke-test onto the instance and run it**

From the local machine (in a second terminal):
```bash
scp -P <VAST_SSH_PORT> vast-templates/fusion-llama3-8b/smoke-test.sh \
    root@<VAST_HOST>:/workspace/smoke-test.sh
```

On the instance:
```bash
bash /workspace/smoke-test.sh
```

Expected output (after the worker finishes loading weights, typically 10–15 min):
```
  ✓ tmux session: controller
  ✓ tmux session: worker
  ✓ tmux session: api
  ✓ tmux session: webui
  ✓ controller /list_models HTTP 200
  ✓ controller lists fusion-llama3-8b
  ✓ API /v1/models HTTP 200
  ✓ API lists fusion-llama3-8b
  ✓ WebUI /health HTTP 200
ALL CHECKS PASSED
```

- [ ] **Step 5: Open `http://localhost:3000` on the laptop**

Sign up admin, pick `fusion-llama3-8b`, send "hello" — expect a coherent text response within a few seconds.

- [ ] **Step 6: Image smoke test**

Upload a small image (a JPEG under ~500 KB) and ask "what's in this image?". Two possible outcomes:

- **Coherent response** → the whole pipeline including vision works; no extra work needed.
- **Garbage / error / model appears to ignore the image** → the FastChat OpenAI adapter isn't forwarding the image field to the FUSION worker. Apply the vision-shim mitigation from spec §7 — open a new brainstorming round to design the shim (out of scope for this plan).

- [ ] **Step 7: Report status back**

Either (a) mark the plan complete, or (b) open follow-up issue(s) for any failed checks.

---

## Self-review notes

Spec coverage check (each spec section → task):

| Spec §                         | Implemented in |
|--------------------------------|----------------|
| §4 Architecture                | Tasks 2–7 (script stages wire the 4 components) |
| §5.1 Stage 1 env setup         | Task 2         |
| §5.2 Stage 2 FUSION + FastChat | Task 3         |
| §5.3 Stage 3 Open WebUI venv   | Task 4         |
| §5.4 Stage 4 controller+worker | Task 5         |
| §5.5 Stage 5 OpenAI API        | Task 6         |
| §5.6 Stage 6 Open WebUI        | Task 7         |
| §6 Ports & SSH tunnel          | README (Task 10) |
| §7 Risks & mitigations         | README (Task 10) + design decisions embedded in Tasks 3, 5 |
| §8 Step-by-step Vast guide     | README (Task 10) |
| §9 Success criteria            | smoke-test.sh (Task 9) + Task 11 manual steps |
| §10 Out of scope               | not implemented (as intended) |

No placeholders, no "TBD", no "similar to task N" — every task contains its own concrete code blocks.

Type/name consistency:
- `fusion-llama3-8b` is used consistently as the `--model-name`, the grep target in guards, and the smoke-test expected string.
- `$SYS_PY`, `$WEBUI_VENV`, `$LOGS`, `$HF_HOME` defined once in Stage 1 and reused thereafter.
- Ports 21001/21002/8000/3000 identical across script, smoke-test, README, and spec.
