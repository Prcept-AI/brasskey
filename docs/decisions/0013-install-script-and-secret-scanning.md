# 0013: Install script and secret scanning

Date: 2026-11-08
Status: accepted

## Context

v0.1 promises one-line installs. `pip install brasskey` and `uv tool install brasskey` work
since 0.1.0a1 (ADR 0012). The install script is next; Homebrew and Docker follow. Separately,
the maintainers asked for a deep credential check before every commit: a leaked key in a public
repository or a published sdist can't be taken back.

## Decision

- **`scripts/install.sh`** (POSIX sh, tested under macOS `sh` and Ubuntu's `dash`):
  - It uses an existing uv. Otherwise it downloads the uv release (0.13.6, the version CI pins)
    for the machine and checks it against a SHA-256 written in the script before running
    anything. Rosetta shells on Apple Silicon get the native build, and musl Linux is detected.
  - It then runs `uv tool install --upgrade brasskey` (`BRASSKEY_VERSION` pins a version).
  - It touches only the user's account: no sudo, and it never edits shell profiles; it prints
    the PATH line instead.
  - It runs only when completely downloaded, because everything is in `main` and called on the
    last line.
  - A test keeps its uv version and checksums in step with CI's uv version.
  - Verified for real: a clean Linux account with no uv got uv 0.13.6 (checksum-verified) and
    Brasskey 0.1.0a1 from PyPI.
  - Published at the raw GitHub URL of `main`, which works once the repository is public; a short
    URL on the project website can redirect there later.
- **Secret scanning:**
  - `scripts/check_secrets.sh` runs gitleaks (MIT, v8.30.1, local or the Docker image pinned by
    digest) on the working tree and on every commit of every branch and tag.
  - `.gitleaks.toml` adds reviewed allowlists only, each narrow: one path AND one value pattern.
  - `.gitleaksignore` records reviewed historical findings by fingerprint: the made-up test keys
    in the first commit.
  - Tests generate keys at run time (`fake_api_key()`), so no key-like string is committed.
  - A repository-policy test, run in CI, fails on common token formats (cloud, GitHub, PyPI,
    Hugging Face, OpenAI and Anthropic, Slack, Google, private keys, hard-coded `bk-` keys) and
    on tracked files that usually hold secrets (`.env`, `.pem`, `.key`, `.netrc`, `.pypirc`).
  - First full audit (2026-11-08): gitleaks, detect-secrets and the pattern search over all
    history and the working tree found no real credential. The only findings were the made-up
    test keys and a metric name.

## Consequences

- No new project dependencies. gitleaks runs as an external tool, like zizmor and pip-audit.
- Bumping uv means updating the script's version and its six checksums together; a test catches
  a mismatch with CI.
- Later: a gitleaks job in CI (the image pinned by digest), and GitHub secret scanning with push
  protection once the repository is public.
