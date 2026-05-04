# Vast Template Fields — Qwen3-VL 8B (vastai/vllm base)

This file is the canonical source for the values you paste into the Vast.ai
Console template editor. Open it side-by-side with the Vast Console while
configuring the template.

> Spec: `docs/superpowers/specs/2026-05-03-vastai-vllm-template-migration-design.md`
> Onstart source: `vast-templates/qwen3-vl-8b/onstart.sh`

## 1. Identification

| Field | Value |
|---|---|
| Template Name | `Open WebUI - Qwen3-VL 8B (vastai/vllm)` |
| Template Description | `Qwen3-VL 8B served via vastai/vllm pre-built (no install). Open WebUI auto-installed via onstart, accessed via SSH tunnel (port 3000). Includes image-aware context compressor filter.` |

## 2. Docker Repository And Environment

| Field | Value |
|---|---|
| Image Path:Tag | `vastai/vllm:v0.20.0-cuda-13.0` |
| Version Tag | `v0.20.0-cuda-13.0` (pick from dropdown) |

## 3. Docker Options (port mapping)

```
-p 1111:1111 -p 7860:7860 -p 8080:8080 -p 8265:8265
```

| Port | Why mapped | Why NOT mapped |
|---|---|---|
| 1111 | Instance Portal | — |
| 7860 | image's vLLM web UI | — |
| 8080 | image's Jupyter | — |
| 8265 | image's Ray dashboard | — |
| 8000 | — | vLLM API uses `sk-dummy`; SSH-tunnel only |
| 3000 | — | Open WebUI binds 127.0.0.1; SSH-tunnel only |

## 4. Environment Variables

Paste each row as a Key/Value pair. Order does not matter.

| Key | Value |
|---|---|
| `OPEN_BUTTON_PORT` | `1111` |
| `OPEN_BUTTON_TOKEN` | `1` |
| `JUPYTER_DIR` | `/` |
| `DATA_DIRECTORY` | `/workspace/` |
| `PORTAL_CONFIG` | `localhost:1111:11111:/:Instance Portal\|localhost:7860:17860:/:vLLM UI\|localhost:8080:18080:/:Jupyter\|localhost:8265:18265:/:Ray Dashboard\|localhost:3000:13000:/:Open WebUI` |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` |
| `VLLM_ARGS` | `--max-model-len 32768 --gpu-memory-utilization 0.90 --trust-remote-code --dtype float16 --served-model-name qwen3-vl-8b --enforce-eager --download-dir /workspace/.hf_cache` |
| `AUTO_PARALLEL` | `false` |
| `HF_HOME` | `/workspace/.hf_cache` |
| `OPENWEBUI_ENABLE` | `true` |

> **Do not include `RAY_ADDRESS` or `RAY_ARGS`** — those are for multi-GPU
> tensor parallel, which we explicitly disable via `AUTO_PARALLEL=false`.

> **Do not include `--mm-encoder-attn-backend TORCH_SDPA` in `VLLM_ARGS`** —
> vLLM 0.20 (shipped in `vastai/vllm:v0.20.0-cuda-13.0`) does not recognize
> this flag and the engine will fail to initialise. The image's earlier
> templates carried this flag from a vLLM 0.19-era config; it is not needed
> for Qwen3-VL on 0.20 and was removed during testing on 2026-05-04.

> **`--enforce-eager` IS required for Qwen3-VL multimodal on vLLM 0.20.**
> Without it, vLLM's CUDA graph captures a fixed-size deepstack buffer (~82
> tokens) at compile time. Real images often produce more deepstack tokens
> (we observed 88 for a 64×64 test PNG) → engine crashes with
> `ValueError: Requested more deepstack tokens than available in buffer` at
> `qwen3_vl.py:1711`. `--enforce-eager` disables CUDA graph compilation so
> the deepstack buffer is sized dynamically per request. Trade-off: ~10–20%
> throughput penalty vs. CUDA graph mode, but the only known-working setup
> for Qwen3-VL multimodal on this vLLM version.

> **Note on `VLLM_ARGS` propagation:** `onstart.sh` launches vLLM with a
> hardcoded set of flags (the same listed above). Setting `VLLM_ARGS` here
> only matters if/when you switch the template to the modern entrypoint
> mode (where the image's `supervisord` invokes `vllm serve $VLLM_MODEL
> $VLLM_ARGS`). Keep it in sync as a future-compatible reference.

## 5. On-start Script

Paste the entire contents of `vast-templates/qwen3-vl-8b/onstart.sh` into the
"On-start Script" textarea. Do not modify it inline in the Vast UI — keep
edits in the repo file so changes are version-controlled.

## 6. Common pitfalls

- **`PORTAL_CONFIG` literal `|` (pipe) characters**: in Vast UI's env-var input
  they are literal pipes, not escaped. The markdown table above shows them as
  `\|` only because that's how the table renders in markdown — paste the raw
  pipe `|` into Vast.
- **Quotes around values**: do NOT add surrounding `"..."` to env-var values.
  The Vast UI handles quoting itself; extra quotes become part of the value.
- **`VLLM_ARGS`**: a single long string with space-separated flags, all on one
  line. Do not split across lines.
- **GPU offer**: when renting from this template, pick a 24 GB+ GPU (RTX 3090,
  4090, A5000, A6000). Smaller GPUs cannot fit Qwen3-VL-8B at fp16 + 32k
  context.

## 7. Verification after save

When you click "Save" in Vast Console:
- Template appears in your Templates list with the new name.
- "Use" button on the template card lets you rent an instance from it.
- The previous template is unaffected (you cloned it; original retained for
  rollback unless you explicitly deleted it).
