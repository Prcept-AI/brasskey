# 0014: Container image

Date: 2026-11-08
Status: accepted (built and tested locally and in CI; first push with the next release tag)

## Context

v0.1 promises a Docker install channel. A container is mainly useful for checking and
benchmarking servers from CI pipelines and other machines (`doctor --security <url>`,
`bench <url>`). Running engines from inside it would need the Docker socket or the engines
themselves, so that isn't its job.

## Options

- **How to build:** Docker's GitHub Actions (setup-qemu, setup-buildx, login, build-push), or
  plain `docker buildx`, which GitHub's runners already have. The plain commands add no
  third-party actions to the supply chain.
- **What goes in:** install from PyPI inside the build, or the exact release wheel plus the
  hash-locked requirements.

## Decision

- **`docker/Dockerfile`:**
  - Base `python:3.12-slim`, pinned by digest, with uv 0.13.6 copied from Astral's image (pinned
    by digest) so `brasskey bench` can set up GuideLLM.
  - Dependencies are installed with `pip --require-hashes` from the requirements exported from
    `uv.lock`, then the release wheel with `--no-deps`.
  - It runs as an unprivileged user (uid 10001), with `ENTRYPOINT ["brasskey"]` and OCI labels.
  - The build context holds only the wheel and `requirements.txt`, never the repository, so
    nothing else can end up in the image.
- **Release (`release.yml`, job `container`, after PyPI succeeds):**
  - `docker buildx` with the docker-container driver: BuildKit v0.34.1 and binfmt qemu-v10.2.3,
    both pinned by digest.
  - Builds linux/amd64 and linux/arm64, with `--provenance=mode=max`.
  - Pushes `ghcr.io/prcept-ai/brasskey:<version>`, plus `latest` for final versions only. The job
    gets `packages: write` and logs in with the workflow token.
  - BuildKit's own image SBOM is left out: it pulls another, unpinned scanner image. The
    release's SPDX SBOM covers the Python packages.
- **CI (`ci.yml`, job `container`):** on every push, builds the amd64 image and runs it:
  - `--version` and `doctor --json`;
  - the user isn't root;
  - `pip check`.
- **Policy test:** every image in the Dockerfile and in workflow image settings is pinned by
  digest.

## Consequences

- Image size is about 264 MB (Python and the dependencies). The first `brasskey bench` inside a
  container downloads GuideLLM (about 300 MB), so a volume on `/home/brasskey/.cache` keeps it.
- Verified here: a local multi-arch build with the pinned BuildKit, and the arm64 image run as
  the CI job runs it (version, doctor, uid 10001, pip check).
- The package on ghcr.io must be made public once, by hand, when the repository goes public.
- Dependabot proposes base-image digest updates weekly, after the same 7-day cooldown, because
  the `FROM` lines are plain and pinned.
