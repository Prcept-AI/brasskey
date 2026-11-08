# 0010: vLLM and SGLang adapters

Date: 2026-11-08
Status: accepted (verified with fakes and the real Docker CLI; real-GPU verification pending on
AWS, see docs/gpu-testing.md)

## Context

Milestone (g) adds the two NVIDIA engines. They differ from llama.cpp and MLX in ways that shape
the adapters:

- They need Linux, an NVIDIA GPU and a driver that matches their CUDA build.
- Their environments pin PyTorch and CUDA (gigabytes), so Brasskey doesn't install them.
- Both publish official Docker images, which is how most teams run them.
- Both reserve most of the GPU memory at start and batch requests inside it.

The engines and their images change often, so both are pinned to checked versions.

## Options

- **Runtime:** process only (the user installs the engine), Docker only, or both.
- **Pinning:** follow `latest`, pin by tag, or pin by digest.
- **API keys:** the engine's own key, or the Brasskey proxy only.
- **Concurrency:** cap the engine at the planned concurrency, or let it batch.

## Decision

- **Versions.**
  - vLLM 0.32.0 (2026-10-18).
  - SGLang 0.5.22 (2026-10-19). SGLang 0.5.23 was 6 days old, inside the 7-day cooldown.
  - Both are CUDA 13.0 builds, so the minimum compute capability for vLLM rises from 7.0 to 7.5:
    CUDA 13 dropped Volta.
- **Runtimes,** detected in order. Detection is skipped off Linux or without an NVIDIA driver
  (`/proc/driver/nvidia`, `/dev/dxg` on WSL2, or `nvidia-smi`).
  1. **Process:** the engine's console script on PATH (`vllm`, or `sglang` run as
     `<its python> -m sglang.launch_server`). Versions and torch's CUDA version are read by the
     engine's own interpreter from package metadata, without importing torch.
  2. **Docker,** when the daemon answers and the NVIDIA Container Toolkit is present. The images
     are pinned by multi-architecture digest:

     | Engine | Image | Pull size |
     | --- | --- | --- |
     | vLLM | `vllm/vllm-openai:v0.32.0@sha256:befccf0e…` | 9.2 GB |
     | SGLang | `lmsysorg/sglang:v0.5.22-runtime@sha256:c3d99f07…` | 14.4 GB |

     The pull is a one-time setup, shown and confirmed like a download (timeout 3 h).
- **Containers.** Built in `engines/docker.py`:
  - `docker run --rm --pull never` (never a surprise download);
  - only the GPUs the plan chose (`--gpus "device=…"`);
  - the engine port published on `127.0.0.1` only;
  - the model's Hugging Face cache folder mounted read-only (its snapshot links resolve inside
    the mount);
  - `--security-opt no-new-privileges` and a `brasskey.managed=true` label;
  - `--shm-size 16g` for multi-GPU;
  - vLLM runs as the image's non-root user (`2000:0`); the SGLang image has none, so it keeps
    root;
  - SGLang keeps NVIDIA's entrypoint, which sets up forward-compatible CUDA.

  Containers are stopped with `docker stop --time 20` (a new `LaunchSpec.stop_argv`, run by the
  supervisor before signals), because signals reach only the Docker client.
- **Launch settings.**
  - Explicit context length (`--max-model-len` / `--context-length`) and tensor-parallel size
    (from the plan's GPU count).
  - GPUs: the ones with the most free memory, passed as `CUDA_VISIBLE_DEVICES` with
    `CUDA_DEVICE_ORDER=PCI_BUS_ID` for processes, or `--gpus` for containers.
  - vLLM gets `--gpu-memory-utilization 0.90`, the share the plan assumed. SGLang sizes its
    static share itself, which depends on the GPU and its CUDA graphs, so 85% stays a planning
    estimate.
  - bf16 weights on GPUs without bfloat16 (before Ampere, e.g. T4) run as float16
    (`--dtype half`).
  - Both engines batch freely inside their reservation; Brasskey doesn't cap concurrent
    sequences. Their memory is preallocated anyway, and a cap of 1 would serialize every request.
- **Privacy.** Request logging stays off in both: Brasskey never passes `--enable-log-requests`
  or `--log-requests`. vLLM's uvicorn access log is off. Telemetry variables are set as for every
  engine.
- **Keys.**
  - vLLM gets the per-run internal key through `VLLM_API_KEY` (`-e VLLM_API_KEY` by name in
    Docker). vLLM checks it only on `/v1`-style paths, so the engine port stays private on
    loopback.
  - SGLang takes a key only as a command-line argument, so it gets none; the proxy enforces keys,
    as for MLX (ADR 0009).
- **NVIDIA checks.**
  - Compute capability, per engine and per quantization (`check_engine`, `check_quantization`).
  - GPUs with MIG enabled are refused for these engines.
  - A driver older than the engine's CUDA version is a warning, not a refusal: datacenter GPUs
    can use forward compatibility, which Brasskey turns on for the vLLM image
    (`VLLM_ENABLE_CUDA_COMPATIBILITY=1`). The warning names the driver to install.
  - VRAM fit comes from the planner (ADR 0004).
- **Failures.** The log tail is matched against shared rules (`engines/gpu_failures.py`):
  - old driver;
  - missing toolkit;
  - Docker permission;
  - missing image;
  - unreadable model files;
  - GPU memory used by others;
  - context too long for the KV cache;
  - out of memory;
  - bfloat16;
  - NCCL;
  - port in use;
  - unsupported model.

  The Docker messages were checked against the real Docker CLI.
- **Locks.** `LockedEngine` gains `runtime` and `image`. Machine-specific argument text (the
  model path, the cache folder) becomes a placeholder (`{model}`, `{model_cache}`) through
  `LaunchSpec.placeholders`, so locks still hold no absolute paths.

## Consequences

- With the GPU engines' prerequisites in place (an NVIDIA driver plus Docker and the toolkit, as
  on AWS GPU images), the first `brasskey run` on that machine offers vLLM by default, and the
  pull size is part of the confirmation.
- Real-GPU behavior is unverified until `uv run pytest -m gpu` runs on AWS. The riskiest
  assumptions are vLLM as UID 2000, `--shm-size` instead of `--ipc=host`, and SGLang's automatic
  memory share on small GPUs.
- Container caches (torch.compile, Triton) are discarded with each `--rm` container, so every
  start recompiles. A cache volume is a later improvement.
- If Brasskey is killed with SIGKILL, the container keeps running, like a process engine (ADR
  0005). Its label lets a future `brasskey stop` or `doctor` find it.
- Upgrading an engine means updating the version, digest, CUDA version and size together, then
  running the GPU tests.
