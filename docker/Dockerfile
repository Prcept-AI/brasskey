# SPDX-License-Identifier: Apache-2.0
#
# Brasskey CLI image: ghcr.io/prcept-ai/brasskey:<version> (ADR 0014).
#
# Built by the release workflow from a context holding only the release wheel and the
# hash-locked runtime requirements (never the repository), so nothing else can end up inside:
#
#   uv export --frozen --no-dev --no-emit-project --format requirements.txt -o ctx/requirements.txt
#   cp dist/brasskey-*.whl ctx/
#   docker build -f docker/Dockerfile -t brasskey ctx
#
# Base images are pinned by digest. The image runs as an unprivileged user; uv is included so
# `brasskey bench` can set up its pinned GuideLLM (mount a volume on /home/brasskey/.cache to keep
# it between runs).

# Plain FROM lines with digests, so Dependabot can keep them current (.github/dependabot.yml).
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

LABEL org.opencontainers.image.title="Brasskey" \
      org.opencontainers.image.description="Run, benchmark and secure open models on your own hardware." \
      org.opencontainers.image.source="https://github.com/Prcept-AI/brasskey" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Prcept AI LLP"

COPY --from=uv /uv /uvx /usr/local/bin/
COPY requirements.txt brasskey-*.whl /tmp/brasskey/

# Dependencies only from the lock (every file checked against its hash), then the wheel itself.
RUN python -m pip install --no-cache-dir --no-deps --require-hashes \
        --requirement /tmp/brasskey/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps /tmp/brasskey/brasskey-*.whl \
    && rm -rf /tmp/brasskey \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin brasskey

USER brasskey
WORKDIR /home/brasskey
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["brasskey"]
CMD ["--help"]
