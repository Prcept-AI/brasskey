# Changelog

Notable changes to Brasskey, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/) (`0.1.0a2` is the second alpha of 0.1.0).

## [Unreleased]

### Security

- `brasskey doctor --security` reported some open model servers as key-protected. It probed with
  a foreign web page's `Origin` header only, so a server that refuses web pages but needs no key
  (Ollama with its default origins, for example) looked protected, even when it listened on the
  network. It now asks twice: like a script (does it need a key?) and like a web page (can any
  website read its answers?).

### Changed

- `brasskey run --dry-run --generate-api-key` no longer creates or saves a key: the plan says a
  new key is created when the server starts, and `Next:` keeps `--generate-api-key`. A dry run
  never changes the keychain.
- A new key that replaces one saved in the keychain says so, because clients with the old key
  are refused from then on.
- `brasskey bench` includes each run's resolution in its intervals. Throughputs are rates of
  finished requests, so runs that finished the same number of requests no longer claim an
  interval of zero width; longer runs (`--duration`) narrow it.
- A llama.cpp build without a GPU backend (no CUDA on an NVIDIA machine, say) is planned for
  system memory and started with no GPU layers, with a warning on how to get a GPU build.
  Brasskey reads this from the build's own `--list-devices`.
- On NVIDIA machines, the llama.cpp install hint builds the tested version (b12110) with CUDA
  into `~/.local/bin`, where Brasskey finds it. The Docker image it suggested before was not
  something Brasskey could use.

### Fixed

- `brasskey run` checks the port before asking to confirm and before downloading, so a busy
  port no longer costs a download first. `Next:` suggests the same command with a free port.
- Without a terminal to confirm in, the error's `Next:` is the same command with `--yes`.
- `brasskey doctor` names engines by their display names ("llama.cpp").

## [0.1.0a2] - 2026-11-08

### Added

- Install channels:
  - Homebrew: `brew install prcept-ai/tap/brasskey`, with bottles for macOS 15 on Apple
    Silicon and Linux (x86_64 and arm64), built in the tap's CI with build provenance.
  - A one-line installer (`scripts/install.sh`) that sets up uv from checksum-verified
    downloads, then Brasskey.
  - A container image, `ghcr.io/prcept-ai/brasskey`, for linux/amd64 and linux/arm64. It runs
    as an unprivileged user, installs dependencies only from hash-locked requirements, and
    carries build provenance.
- Python 3.15 support.
- GitHub build provenance for the wheel and sdist, now that the repository is public.

## [0.1.0a1] - 2026-11-08

The first release.

### Added

- `brasskey run`: detects the hardware, picks an engine and a model that fits (or plans the one
  you name), shows the download size and time, asks before heavy work, and serves an
  OpenAI-compatible API on 127.0.0.1 through Brasskey's local proxy.
- `brasskey bench`: benchmarks an endpoint with GuideLLM over repeated runs, reports 95%
  intervals, and compares two endpoints (or saved results) with a PASS, FAIL or INCONCLUSIVE
  verdict.
- `brasskey doctor`: hardware, engines and running servers; `--security` checks addresses,
  keys, TLS and exposure on this machine or for one URL.
- Read-only hardware probes for NVIDIA GPUs (NVML), Apple Silicon and x86/ARM CPUs.
- Engine adapters for llama.cpp, MLX, vLLM and SGLang. vLLM and SGLang are tested with fake
  GPUs so far; tests on real NVIDIA GPUs come next.
- Secure defaults: loopback only, an API key for any other address, optional TLS, keys from
  environment variables or the OS keychain only, telemetry off, no prompts logged.
- `brasskey.yaml` (intent) and `brasskey.lock` (resolved plan and measurements), with JSON
  Schemas.
- Every command supports `--json` and ends with a `Next:` suggestion.
- Releases through PyPI trusted publishing, with Sigstore attestations, an SPDX SBOM, checksums
  and reproducible builds.

[Unreleased]: https://github.com/Prcept-AI/brasskey/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/Prcept-AI/brasskey/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/Prcept-AI/brasskey/releases/tag/v0.1.0a1
