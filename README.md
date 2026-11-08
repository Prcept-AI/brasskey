<br>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/brasskey-wordmark-on-dark.svg">
    <img alt="Brasskey" src="brand/logo/brasskey-wordmark-on-light.svg" width="240">
  </picture>
</p>

<p align="center"><b>The key to every engine.</b></p>

<br>

Brasskey is an open-source tool to deploy, tune, verify and serve open models on any hardware,
measured on your own traffic.

```bash
brasskey run        # try any model on your machine
brasskey switch     # prove an open model can replace an expensive API (coming soon)
brasskey deploy     # ship it tuned, verified and secure (coming soon)
```

## Status

Pre-alpha, building v0.1 "It runs": `brasskey run`, `brasskey bench` and
`brasskey doctor --security` on NVIDIA GPUs, Apple Silicon and CPUs, with llama.cpp, MLX, vLLM and
SGLang. Done so far:

- `brasskey doctor`: hardware, engines and running servers.
- Planning with `brasskey run --dry-run`.
- `brasskey run` with llama.cpp, MLX (Apple Silicon), vLLM or SGLang (NVIDIA on Linux, as a
  process or from the official image pinned by digest). It downloads and verifies the model,
  then serves it on 127.0.0.1 through a Brasskey proxy that enforces API keys. Ctrl-C stops it
  cleanly.
- `brasskey doctor --security`: checks this machine, or a URL, for exposed model servers.
- `brasskey bench`: repeated GuideLLM runs, 95% intervals on every metric, and PASS / FAIL /
  INCONCLUSIVE when comparing two setups.

Next: release preparation. Design decisions are recorded in [docs/decisions](docs/decisions/).

## Try it from source

You need [uv](https://docs.astral.sh/uv/) 0.13.6 or newer.

```bash
git clone https://github.com/Prcept-AI/brasskey.git
cd brasskey
uv sync
uv run brasskey --help
uv run brasskey doctor          # what Brasskey sees on this machine (CPU, memory, GPUs)
uv run brasskey doctor --json
uv run brasskey run --dry-run           # the model Brasskey would pick for this machine, and why
uv run brasskey run qwen3-14b --dry-run --context 32768
uv run brasskey run hf:Qwen/Qwen3-4B-GGUF:Q8_0 --dry-run   # any Hugging Face repo

brew install llama.cpp                  # the engine (macOS or Linux)
uv run brasskey run qwen3-0.6b          # download (639 MB), start, serve; Ctrl-C stops it
uv run brasskey run qwen3-4b --save     # also writes brasskey.yaml + brasskey.lock
uv run brasskey run --engine mlx        # Apple Silicon: MLX via uv, set up on first use
uv run brasskey run --engine vllm       # NVIDIA on Linux: vLLM on PATH, or its pinned image

uv run brasskey bench http://127.0.0.1:8000/v1 --save base.json      # measure a server
uv run brasskey bench http://127.0.0.1:8001/v1 --baseline base.json  # PASS / FAIL / INCONCLUSIVE
uv run brasskey bench http://127.0.0.1:8001/v1 --compare http://127.0.0.1:8000/v1

uv run brasskey doctor --security       # exposed model servers, keys, Docker, firewall
uv run brasskey doctor --security http://gpu-box:8000/v1   # check a server you run elsewhere
uv run brasskey run qwen3-4b --host 0.0.0.0 --generate-api-key   # reachable from your network
```

Every command supports `--json` (one JSON document on stdout), `--debug` (logs and full
tracebacks on stderr) and `--yes` (skip confirmations), and ends with a `Next:` suggestion.

## Secure and private by default

Servers bind to `127.0.0.1`. Any other address requires a strong API key (from
`BRASSKEY_API_KEY` or your OS keychain, never a command-line argument). There is no telemetry,
and prompt content is never logged. See [SECURITY.md](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## License and pledge

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). **The Brasskey core is Apache 2.0 and
stays Apache 2.0.** Paid features for organizations live in separate repositories and never
change the license of this one. The Brasskey name and logo are trademarks of Prcept AI LLP; see
[TRADEMARKS.md](TRADEMARKS.md).

Made by Prcept AI LLP.
