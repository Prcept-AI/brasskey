#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Scan for credentials before every commit, push and release:
#
#   sh scripts/check_secrets.sh
#
# Runs gitleaks (MIT) twice with the repository's rules (.gitleaks.toml, .gitleaksignore):
#   1. the working tree, including files not committed yet;
#   2. every commit on every branch and tag (a secret deleted later is still in history).
# Uses a local `gitleaks` if installed (brew install gitleaks), otherwise its Docker image pinned
# by digest. Exits 1 on any finding; findings are printed with the secret redacted.
#
# If something real is found: don't commit it. Remove it, and revoke and rotate the credential,
# because anything that reached a commit or a remote must be treated as leaked.

set -eu

IMAGE="zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

run_gitleaks() {
    if command -v gitleaks >/dev/null 2>&1; then
        (cd "$ROOT" && gitleaks "$@")
    elif command -v docker >/dev/null 2>&1; then
        docker run --rm -v "$ROOT":/repo:ro -w /repo "$IMAGE" "$@"
    else
        echo "check_secrets: install gitleaks (brew install gitleaks) or start Docker." >&2
        exit 2
    fi
}

main() {
    status=0
    echo "check_secrets: working tree" >&2
    run_gitleaks dir . --config .gitleaks.toml --redact --no-banner || status=1
    echo "check_secrets: history (all branches and tags)" >&2
    run_gitleaks git . --log-opts=--all --config .gitleaks.toml --redact --no-banner || status=1
    if [ "$status" -ne 0 ]; then
        echo "check_secrets: FAILED: possible credentials found (see above). Do not commit." >&2
        exit 1
    fi
    echo "check_secrets: OK: no credentials found" >&2
}

main "$@"
