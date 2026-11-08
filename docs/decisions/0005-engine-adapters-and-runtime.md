# 0005: Engine adapter contract and process runtime

Date: 2026-11-08
Status: accepted

## Context

Milestone (d) makes `brasskey run` start a real engine (llama.cpp). MLX, vLLM and SGLang follow in
(e) and (g), so the shape chosen now is the one every engine and community plugin will implement.
The project rules (CONTRIBUTING.md) require engines to run outside the Brasskey process, and engine
churn (frequent releases that change flags and behavior) is the likeliest way for adapters to break.

## Options

- **Adapter scope:** each adapter owns its whole lifecycle (spawn, wait, stop), or adapters stay
  small (detect, install hint, launch spec, failure explanation, health) and one shared
  supervisor owns processes.
- **Version churn:** pin one engine version per Brasskey release, or detect what the installed
  version supports.
- **Secrets:** API key on the command line (simple), or via the engine's environment.
- **Foreground lifecycle:** engine in Brasskey's process group (Ctrl-C hits both at once), or its
  own session, with Brasskey forwarding the stop.

## Decision

- **Adapter contract** (`engines/base.py`):
  - `detect() -> EngineInstall | None` reports version, executable, supported flags, and devices
    with their memory.
  - `install_hint(hw)` gives the exact command to install the engine.
  - `build_launch(install, model, options, hw) -> LaunchSpec` gives argv, non-secret env, secret
    env *names*, URLs and a startup timeout.
  - `explain_failure(exit_code, log_tail)` turns the engine's log into a plain-language error.
  - `health()` defaults to an HTTP 200 check.

  Hardware support rules stay in `engines/capabilities.py`. The registry (`engines/registry.py`)
  lists the engines this build can start, and the planner only chooses among those.
- **Version-proof launches:** llama.cpp's `--help` is parsed once at detection, and optional
  flags (`--jinja`, `--metrics`, `--no-webui`, `--offline`, `--fit off`) are added only when
  supported. Sizing flags (context, GPU layers, slots) are always explicit, so llama.cpp's
  automatic `--fit` can't change a plan silently.
- **Defaults for llama.cpp:**
  - API only (`--no-webui`), Prometheus metrics on the same local port, and no downloads by the
    engine (`--offline`).
  - Every engine also gets `DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`
    and `VLLM_NO_USAGE_STATS=1`.
- **Supervisor** (`runtime/supervisor.py`):
  - The engine runs in its own session and process group. Its output goes to a 0600 log in
    Brasskey's log folder.
  - Secrets are copied from Brasskey's environment by variable name (llama.cpp:
    `LLAMA_API_KEY`), never put in argv.
  - Health is polled until ready. If the engine exits, reports a failure or times out, the log
    tail is explained in plain language.
  - Stop sends SIGTERM to the group, then SIGKILL after 10 s. Children die with it.
- **Foreground:** `brasskey run` serves until Ctrl-C or SIGTERM, then stops the engine.
  `--json` prints one "ready" document and nothing else on stdout.
- **State files** (`runtime/state.py`): one private JSON file per running server. A record
  counts as live only if the process start time also matches (guards against process-ID reuse).
  Stale records are removed when listed. `doctor` shows running servers;
  `doctor --security` (f) will use them.
- **Downloads** (`models/fetch.py`):
  - Files come only from the pinned revision.
  - Fresh files are hashed against the catalog's SHA-256 before an engine sees them. Cached
    files are accepted when the Hugging Face cache stores them under their SHA-256 name.
  - Progress comes from huggingface_hub's `tqdm_class` hook. Only bars sized like the file
    count, which avoids double counting with Xet downloads.
- **Measurement over estimate:**
  - llama.cpp's `--list-devices` reports Metal's working-set limit, which replaces the macOS
    estimate from ADR 0003 (12288 MiB on the M3 Pro, matching the estimate).
  - The same list shows which GPU backends the build has. A build that lists no GPU (a CPU-only
    llama.cpp on an NVIDIA machine, say) is planned for system RAM and launched with no GPU
    layers, and the plan warns how to get a GPU build. When the list is missing (an older
    version, or the command failed), Brasskey keeps assuming the machine's GPU.
  - Startup time is measured and stored in the lock.

## Consequences

- New engines plug in with a small adapter and golden-command-line tests. The supervisor, state
  files, downloads and output are shared.
- Brasskey must stop engines itself: if Brasskey is killed with SIGKILL, the engine keeps
  running. The state file then lets `doctor` show it, and the port check refuses to reuse its
  port.
- A process started with SIGINT ignored (a background job in a non-interactive shell) won't stop
  on Ctrl-C. SIGTERM works there, as for any Unix daemon.
- Verified end to end: real llama.cpp 0.9.0 + Qwen3 0.6B on an M3 Pro.
  - Chat requests and `/metrics` worked; `doctor` listed the server.
  - Ctrl-C and SIGTERM stopped the server cleanly, leaving no orphans and no state file.
  - The prompt never reached the engine log (`pytest -m engine`).
