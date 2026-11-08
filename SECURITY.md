# Security policy

Brasskey starts model servers, opens network ports and handles API keys, so we take security
reports seriously and thank everyone who reports responsibly.

## Supported versions

Brasskey is pre-alpha (0.1 pre-releases). Until 1.0, only the latest release receives security
fixes.

## Reporting a vulnerability

Please **do not** open a public issue, discussion or pull request for a security problem.

- **Preferred:** GitHub private vulnerability reporting at
  <https://github.com/Prcept-AI/brasskey/security/advisories/new>
- **Email:** praveen92y@gmail.com with the subject "Brasskey security"

Please include the Brasskey version (`brasskey --version`), your OS and hardware, the engine
involved (llama.cpp, MLX, vLLM or SGLang), steps to reproduce, and the impact you observed.
Never include real API keys, tokens or private prompts in a report.

## What to expect

| Step | Target |
| --- | --- |
| Acknowledge your report | within 3 business days |
| Initial assessment and severity | within 7 days |
| Fix or mitigation | within 90 days, sooner for actively exploited issues |
| Disclosure | coordinated with you; credit in the advisory unless you prefer otherwise |

## Scope

**In scope:** Brasskey's code (CLI, engine adapters, configuration and lock files), its defaults
(bind address, authentication, TLS), how it handles keys and logs, its release artifacts and CI.

**Upstream:** vulnerabilities in the engines Brasskey launches (vLLM, SGLang, llama.cpp, MLX) or
in GuideLLM belong to those projects. Report them upstream; we are glad to help coordinate and may
ship mitigations such as safer defaults.

**Out of scope:** issues that need an already-compromised machine or account, the content a model
generates, and resource exhaustion from running a model on undersized hardware.

## Safe harbor

We will not pursue legal action against good-faith research that follows this policy, avoids
privacy violations and service disruption, and only tests systems you own or are allowed to test.

## Security defaults

Implemented in v0.1 ([ADR 0009](docs/decisions/0009-security-defaults-proxy-and-doctor.md)).
`brasskey doctor --security` checks them, and the rest of this machine's exposure.

- **Loopback by default.** `brasskey run` listens on `127.0.0.1`.
- **A key off-loopback.** Any other address requires a strong API key, or Brasskey refuses to
  start, before downloading anything.
- **Where keys live.** Keys come from `BRASSKEY_API_KEY` (or the variable named by
  `--api-key-env`) or the OS keychain (`--generate-api-key` creates one and stores it there).
  They never appear on command lines, in logs, in state files or in `brasskey.lock`, and log
  output redacts them.
- **One proxy for every engine.** Brasskey's own proxy sits in front of every engine. Engines
  listen on a private loopback port. The proxy checks the key, refuses browser requests from
  other web sites (Origin check), serves only the planned model and limits request size.
  Engines that support keys also get a random per-run internal key, so local programs can't
  bypass the proxy.
- **Containers.** vLLM and SGLang images are pinned by digest and publish their port on
  `127.0.0.1` only. They get the model cache read-only, only the planned GPUs and no new
  privileges, and vLLM runs as a non-root user. API keys reach containers by variable name,
  never on a command line.
- **TLS.** `--tls-cert` and `--tls-key` serve HTTPS. Without TLS off-loopback, Brasskey warns that
  traffic is not encrypted.
- **No telemetry.** Engine telemetry is switched off when Brasskey launches an engine.
- **No prompt logging.** Prompt and response content is never logged or stored by default; the
  proxy keeps no access log. `brasskey bench` keeps numbers only: GuideLLM is told to keep no
  request text, and its raw reports are deleted after parsing.
- **Releases.** Releases are published from GitHub Actions through PyPI trusted publishing (no
  tokens), built reproducibly, signed with Sigstore, and shipped with an SBOM and checksums. CI
  pins every action by commit SHA and every tool by exact version (the build backend by hash).

## Verifying a release

Each release on GitHub carries the wheel, the sdist, `SHA256SUMS` and an SPDX SBOM
(`brasskey-<version>.spdx.json`, the locked runtime dependencies with their licenses).

```bash
# 1. Checksums of the files you downloaded
sha256sum --check --ignore-missing SHA256SUMS
# macOS: shasum -a 256 --check --ignore-missing SHA256SUMS

# 2. GitHub build provenance: built by this repository's release workflow
gh attestation verify brasskey-<version>-py3-none-any.whl --repo Prcept-AI/brasskey

# 3. Rebuild it yourself: builds are reproducible, so the hash matches
git checkout v<version>
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build \
  --build-constraints scripts/build-constraints.txt --require-hashes --no-create-gitignore
sha256sum dist/*
```

PyPI also shows the Sigstore-signed attestations (PEP 740) for each file on the release's
"Download files" page.
