# 0007: MLX adapter

Date: 2026-11-08
Status: accepted

## Context

Milestone (e) adds MLX, usually the fastest engine on Apple Silicon. ADR 0002 decided that
Brasskey runs MLX through an isolated, pinned `uvx` environment after the user confirms. Reading
the real `mlx_lm.server` 0.32.2 (source and `--help`) showed several defaults that don't fit a
private, planned server.

## Findings in mlx_lm.server 0.32.2

- `--allowed-origins` defaults to `*`: any web page open in a browser could call the local server
  and read its responses.
- `--decode-concurrency 32`, `--prompt-concurrency 8` and an unbounded prompt cache (10
  sequences) can allocate far more KV cache than planned.
- No API keys. It warns that it "only implements basic security checks".
- Only the literal model name it was started with maps to the loaded model. A request naming
  any other model makes it try to *load* that model (cached models could load; downloads are
  blocked by `HF_HUB_OFFLINE=1`). `/v1/models` lists every MLX model in the Hugging Face cache.
- The model loads before the port opens, so "connection refused, then `/health` 200" means
  ready. Request bodies and generated text are logged only at DEBUG.

## Decision

- **Where it comes from:** a `mlx_lm.server` already on PATH is used as is. Otherwise Brasskey
  runs it with `uv tool run --managed-python --python 3.12 --exclude-newer 2026-11-01T00:00:00Z
  --from mlx-lm==0.32.2`, which pins mlx-lm and resolves all of its dependencies as of a fixed
  date. Every user of this Brasskey version gets the same environment.
  - Detection runs `--offline`, never downloading anything.
  - If the environment isn't cached, the plan says "set up automatically on first run (about
    70 MB, one time)", the confirmation question includes it, and setup runs only after a yes.
  - The probe also reads Metal's working-set limit through MLX (12,884,918,272 bytes on the
    M3 Pro, matching llama.cpp's figure).
- **Launch defaults:**
  - `--allowed-origins` is set to the server's own address only.
  - Concurrency is set to the planned value, and `--prompt-cache-bytes` is capped at the planned
    KV-cache size.
  - `--log-level INFO`.
  - Optional flags are added only when the version supports them (parsed from `--help` for
    PATH installs; known for the pinned version).
- **Naming:** the engine starts in a private working folder that contains a link named after the
  served model (e.g. `qwen3-0.6b`, pointing at the model files), with `--model qwen3-0.6b`. The
  advertised name, or no name, then uses the loaded model. The supervisor creates the folder
  and removes it on stop; unsafe names are refused.
- **API keys:** refused for MLX with a plain error before anything is downloaded. MLX serves on
  loopback only (ADR 0002). (Superseded in milestone f, ADR 0009: the Brasskey proxy checks keys
  and serves only the planned model, so MLX can use keys and other addresses.)
- **Preference:** on Apple Silicon MLX comes first when available. The automatic pick on an
  M3 Pro 18 GB returns to Qwen3 8B (4-bit MLX, 4.4 GB).

## Consequences

- First use costs a one-time ~70 MB setup (about 590 MB unpacked in uv's cache, ~20 s here).
- Measured on the M3 Pro with Qwen3 0.6B, three runs each:
  - llama.cpp (Q8_0) about 146 tokens/s; MLX (4-bit) 167–204 tokens/s end to end.
  - The quantizations differ, so `brasskey bench` (h) does fair comparisons.
- Remaining weaknesses come from mlx_lm.server itself:
  - other model names still trigger loads;
  - `/v1/models` lists the whole cache;
  - there are no API keys.

  Brasskey's loopback-only binding limits the exposure. A small local proxy would close these
  gaps; see the proposal in ADR 0008. (Closed in milestone f by the proxy, ADR 0009.)
- Bumping mlx-lm means updating the version, the cutoff date and the known flags together, and
  re-running `pytest -m engine`.
