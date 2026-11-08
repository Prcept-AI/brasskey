# Contributing to Brasskey

Thanks for your interest! Brasskey is pre-alpha and moving fast toward v0.1 "It runs". This page
covers how to set up, the rules every change follows, and how contributions are licensed.

## Scope first

We build one milestone at a time (see the Status section of the README). Anything outside the
current milestone waits; something moves earlier only if a design partner needs it and something of
equal size moves later. If you want to propose a bigger change, open an issue or write a short
proposal in `docs/decisions/` before writing code.

## Set up

You need [uv](https://docs.astral.sh/uv/) 0.13.6 or newer (`uv self update`).

```bash
uv sync                      # create .venv with runtime + dev dependencies from uv.lock
uv run brasskey --help       # run the CLI from source
```

uv uses its own managed Python builds for this project (`python-preference = "only-managed"`), so
an active conda or system Python does not leak into the environment.

## Everyday commands

```bash
uv run pytest                                   # unit tests; no GPU needed
uv run pytest -m gpu                            # real NVIDIA tests (opt-in)
uv run pytest -m engine                         # real engine + small model download (opt-in)
uv run ruff check . && uv run ruff format --check .
uv run pyright                                  # strict for src/ and scripts/
uv run python scripts/check_licenses.py         # dependency license policy
```

CI runs all of these on Linux and macOS (Apple Silicon) with Python 3.11 to 3.14.

## Rules every change follows

- **Tests:** every feature ships with tests that pass on a laptop without a GPU. Use fake hardware
  probes and fake engine processes; mark real-hardware tests `@pytest.mark.gpu` (or `engine`,
  `network`).
- **License header:** every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0` (a test
  checks this).
- **Engines stay out of process:** never import vLLM, SGLang, MLX, llama.cpp bindings, PyTorch or
  GuideLLM into Brasskey. They run as subprocesses or containers behind an adapter. Ruff enforces
  this (`TID251`).
- **CLI contract:** every command supports `--json`, ends with a one-line `Next:` suggestion,
  explains errors in plain language with a fix, and shows tracebacks only with `--debug`.
- **Measure, don't guess:** recommendations are backed by a measurement or clearly labeled as an
  estimate.
- **Privacy:** never log or store prompt content, API keys or tokens.
- **Secure defaults:** servers bind to `127.0.0.1`; any other address requires an API key. Keys
  come from environment variables or the OS keyring and never appear on command lines, in logs or
  in lock files. Telemetry is off; nothing leaves the machine without explicit opt-in.
- **Reproducible deployments:** `brasskey.yaml` (intent) and `brasskey.lock` (resolved plan and
  measured results) are Pydantic v2 models with exported JSON Schemas. After changing them, run
  `uv run python scripts/export_schemas.py` and commit `schemas/`.
- **Supply chain:** GitHub Actions are pinned by full commit SHA and tools by exact version, and
  `uv.lock` is committed (see also Dependencies below).
- **Naming:** "Brasskey" in prose, `brasskey` in code and commands. No `bk` alias.

## Dependencies

Ask before adding a dependency (open an issue). It must be Apache-2.0, MIT or BSD compatible;
GPL and AGPL code or dependencies are never accepted, and code is never copied from GPL or AGPL
projects. Record the dependency and its license in `NOTICE`. The lock only resolves releases that
have been public for at least 7 days (`exclude-newer` in `pyproject.toml`).

## Decisions

Record notable decisions as short ADRs in `docs/decisions/NNNN-title.md` using the template there.

## Pull requests

Keep pull requests small and focused, fill in the template checklist, and make sure CI is green.
Write commit messages in the imperative mood ("Add hardware probe for Apple Silicon").

## Contributor License Agreement

Brasskey uses a Contributor License Agreement (CLA) based on the Apache Software Foundation's
individual and corporate CLAs: you keep the copyright in your contribution and grant Prcept AI
LLP a broad license (including patents) to use and distribute it. A CLA bot will ask you to sign
once, on your first pull request.

The CLA bot is not set up yet, so **we cannot merge outside pull requests until it is**. Issues,
bug reports and discussions are very welcome in the meantime.

**Our pledge:** the Brasskey core is licensed under Apache 2.0 and stays Apache 2.0. Paid
features for organizations live in separate repositories and never change the license of this one.

## Conduct and security

Everyone taking part follows our [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems
privately as described in [SECURITY.md](SECURITY.md), never in a public issue.
