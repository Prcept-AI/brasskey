# 0015: Homebrew tap

Date: 2026-11-08
Status: accepted

## Context

v0.1 promises `brew` as an install channel. Brasskey is a Python application, and two of its
dependencies are written in Rust (pydantic-core, hf-xet; cryptography too on Linux). Homebrew
installs Python dependencies from source (`--no-binary=:all:`), so without prebuilt bottles every
install would compile Rust for several minutes. homebrew-core requires a stable release and
notability, so a tap comes first.

## Options

- **Formula resources:** let `brew update-python-resources` re-resolve dependencies from PyPI,
  or generate them from `uv.lock`, which pins the versions CI tests.
- **Shared formulas:** compile everything into the formula, or depend on homebrew-core's
  bottled `pydantic`, `cryptography` and `certifi`, as its own Python formulas do (`llm`, `hf`).
  Those formulas use Python 3.15.
- **Bottles:** none (every user compiles), or a tap with Homebrew's bottle CI.
- **Updates:** Homebrew's daily autobump (it re-resolves from PyPI), or regenerate the formula
  for each Brasskey release.

## Decision

- **Tap repository `Prcept-AI/homebrew-tap`** (public, Apache-2.0). Install with
  `brew install prcept-ai/tap/brasskey`.
- **`scripts/homebrew_formula.py`** (in this repository) writes `Formula/brasskey.rb` for a
  version published on PyPI:
  - The resources are Brasskey's runtime dependencies from `uv.lock`, as sdists pinned by URL and
    SHA-256.
  - Platform markers are evaluated for macOS and Linux with Python 3.15. Linux-only packages
    (jeepney, secretstorage) and the `cryptography` dependency go in one `on_linux` block.
    Windows-only packages and `importlib-metadata` (Python < 3.12) are left out.
  - `pydantic`, `cryptography` and `certifi` come from homebrew-core. `rust` and `pkgconf` are
    build dependencies, for hf-xet.
  - hf-xet is built with `native-tls` instead of `rustls`, the same workaround as homebrew-core's
    `hf` formula: rustls pulls in the bundled aws-lc C code, which doesn't compile inside
    Homebrew's build environment. A local Linux build failed this way before the change.
    Linux adds `openssl@4` for native-tls; macOS uses the system TLS.
  - The test block runs `brasskey --version` and `brasskey doctor --json`.
  - `brew style` and `brew audit --strict --new --online` pass; the audit's only finding is the
    homepage, which returns 404 while this repository is private.
- **Python 3.15:** required by homebrew-core's shared formulas, and now part of Brasskey's CI
  matrix and classifiers. The test suite passes on 3.15.
- **Bottles** come from the tap's test-bot workflow, adapted from `brew tap-new`:
  - built for pull requests on macOS 15 (Apple Silicon; the bottles also install on newer macOS)
    and Linux x86_64 and arm64, in Homebrew's container pinned by digest;
  - published by running the `brew pr-pull` workflow with the pull request number, which uploads
    the bottles to a tap release, records Sigstore build provenance, and pushes the bottle commit;
  - every action pinned by SHA; zizmor is clean at the auditor level, with one documented
    suppression for the matrix container expression.
- **No autobump:** each Brasskey release regenerates the formula from its lock (RELEASING.md).

## Consequences

- The Linux build was checked locally in Homebrew's arm64 container before publishing: the
  first try failed on aws-lc, as described above. macOS bottles are built by the tap's CI.
- hf-xet's patch depends on its file layout. If a future version moves those `Cargo.toml` files,
  `inreplace` fails the build loudly instead of silently producing the wrong thing.
- Intel Macs and systems without a bottle build from source, which needs Rust and takes several
  minutes.
- Updating the formula is a manual step per release: generate, open a pull request, run
  `brew pr-pull`. It could later be automated with a GitHub App token scoped to the tap.
