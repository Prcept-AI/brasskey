#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Install Brasskey for your user account:
#
#   curl -LsSf https://raw.githubusercontent.com/Prcept-AI/brasskey/main/scripts/install.sh | sh
#
# What it does, and nothing else:
#   1. Uses uv if you already have it. Otherwise it downloads uv from its GitHub release and
#      checks the archive against a SHA-256 written in this script before running anything.
#   2. Runs `uv tool install brasskey`: your user account only, no sudo, no edits to your shell
#      profile.
#   3. Tells you how to put Brasskey on your PATH if it isn't, and what to run next.
#
# Settings (environment variables):
#   BRASSKEY_VERSION       a version to install, e.g. 0.1.0a1 (default: the newest)
#   BRASSKEY_INSTALL_DIR   where uv goes when it has to be installed (default: ~/.local/bin)
#   BRASSKEY_UV_URL / BRASSKEY_UV_SHA256   a uv archive to use instead (mirrors); both or neither
#
# The whole script is one function called on the last line, so a partly downloaded script does
# nothing. To update uv: change UV_VERSION and the checksums from
# https://github.com/astral-sh/uv/releases/tag/<version> (the .tar.gz.sha256 files).

set -eu

UV_VERSION="0.13.6"
UV_SHA256_aarch64_apple_darwin="60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472"
UV_SHA256_x86_64_apple_darwin="63ba17fdda55ad1a230d4b37fc6017ffa10c23688c83818bd3bfdf914c652a02"
UV_SHA256_aarch64_unknown_linux_gnu="229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d"
UV_SHA256_x86_64_unknown_linux_gnu="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"
UV_SHA256_aarch64_unknown_linux_musl="819692e404d23e5aae6bc44c0e260f1f5492fd1910ab4693b2c650f926dbe800"
UV_SHA256_x86_64_unknown_linux_musl="5d61f26b6c7213779eda90edc6971ea6b0b796bfb8d49a574c8ad69b59838548"

say() {
    printf 'brasskey-install: %s\n' "$1" >&2
}

fail() {
    say "error: $1"
    exit 1
}

# The uv release target for this machine, e.g. aarch64-apple-darwin.
detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$arch" in
        arm64 | aarch64) arch="aarch64" ;;
        x86_64 | amd64) arch="x86_64" ;;
        *) fail "this processor ($arch) isn't supported; Brasskey runs on x86_64 and arm64." ;;
    esac
    case "$os" in
        Darwin)
            # An Intel shell under Rosetta on an Apple Silicon Mac: install the native uv.
            if [ "$arch" = "x86_64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = "1" ]; then
                arch="aarch64"
            fi
            echo "$arch-apple-darwin"
            ;;
        Linux)
            if ldd --version 2>&1 | grep -qi musl; then
                echo "$arch-unknown-linux-musl"
            else
                echo "$arch-unknown-linux-gnu"
            fi
            ;;
        *) fail "$os isn't supported; Brasskey runs on macOS and Linux (Windows: use WSL2)." ;;
    esac
}

expected_sha256() {
    case "$1" in
        aarch64-apple-darwin) echo "$UV_SHA256_aarch64_apple_darwin" ;;
        x86_64-apple-darwin) echo "$UV_SHA256_x86_64_apple_darwin" ;;
        aarch64-unknown-linux-gnu) echo "$UV_SHA256_aarch64_unknown_linux_gnu" ;;
        x86_64-unknown-linux-gnu) echo "$UV_SHA256_x86_64_unknown_linux_gnu" ;;
        aarch64-unknown-linux-musl) echo "$UV_SHA256_aarch64_unknown_linux_musl" ;;
        x86_64-unknown-linux-musl) echo "$UV_SHA256_x86_64_unknown_linux_musl" ;;
        *) fail "no uv checksum for $1." ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        fail "neither sha256sum nor shasum is available to check the download."
    fi
}

download() {
    if command -v curl >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -fsSL -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget --https-only -q -O "$2" "$1"
    else
        fail "neither curl nor wget is available to download uv."
    fi
}

# Download uv, check it, and put uv and uvx in the install folder. Prints the path to uv.
install_uv() {
    target="$(detect_target)"
    if [ -n "${BRASSKEY_UV_URL:-}" ] || [ -n "${BRASSKEY_UV_SHA256:-}" ]; then
        [ -n "${BRASSKEY_UV_URL:-}" ] && [ -n "${BRASSKEY_UV_SHA256:-}" ] ||
            fail "set both BRASSKEY_UV_URL and BRASSKEY_UV_SHA256, or neither."
        url="$BRASSKEY_UV_URL"
        sha="$BRASSKEY_UV_SHA256"
    else
        url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz"
        sha="$(expected_sha256 "$target")"
    fi
    dest="${BRASSKEY_INSTALL_DIR:-$HOME/.local/bin}"
    work="$(mktemp -d)"
    say "uv not found; installing uv $UV_VERSION ($target) into $dest"
    download "$url" "$work/uv.tar.gz" || { rm -rf "$work"; fail "couldn't download $url"; }
    actual="$(sha256_of "$work/uv.tar.gz")"
    if [ "$actual" != "$sha" ]; then
        rm -rf "$work"
        fail "the uv download doesn't match its checksum (expected $sha, got $actual); nothing was installed."
    fi
    tar -xzf "$work/uv.tar.gz" -C "$work"
    mkdir -p "$dest"
    for tool in uv uvx; do
        found="$(find "$work" -type f -name "$tool" | head -n 1)"
        [ -n "$found" ] || { rm -rf "$work"; fail "the uv archive has no $tool."; }
        cp "$found" "$dest/$tool"
        chmod 755 "$dest/$tool"
    done
    rm -rf "$work"
    echo "$dest/uv"
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
    elif [ -x "${BRASSKEY_INSTALL_DIR:-$HOME/.local/bin}/uv" ]; then
        echo "${BRASSKEY_INSTALL_DIR:-$HOME/.local/bin}/uv"
    fi
}

warn_if_old_uv() {
    version="$("$1" --version 2>/dev/null | cut -d ' ' -f 2)"
    minor="$(echo "$version" | cut -d . -f 2)"
    major="$(echo "$version" | cut -d . -f 1)"
    wanted_minor="$(echo "$UV_VERSION" | cut -d . -f 2)"
    case "$major$minor" in
        *[!0-9]* | "") return 0 ;;
    esac
    if [ "$major" -eq 0 ] && [ "$minor" -lt "$wanted_minor" ]; then
        say "your uv is $version; Brasskey's MLX and bench setup expect uv $UV_VERSION or newer."
        say "if they fail, run: uv self update"
    fi
}

main() {
    uv="$(find_uv)"
    if [ -n "$uv" ]; then
        say "using uv at $uv"
        warn_if_old_uv "$uv"
    else
        uv="$(install_uv)"
    fi

    spec="brasskey"
    if [ -n "${BRASSKEY_VERSION:-}" ]; then
        spec="brasskey==$BRASSKEY_VERSION"
    fi
    say "installing $spec with uv tool install"
    "$uv" tool install --upgrade "$spec" || fail "uv couldn't install $spec (see above)."

    bin_dir="$("$uv" tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
    case ":$PATH:" in
        *":$bin_dir:"*)
            say "done. Next: brasskey doctor"
            ;;
        *)
            say "done. $bin_dir isn't on your PATH yet; add it with:"
            say "  export PATH=\"$bin_dir:\$PATH\"   (or run: $uv tool update-shell)"
            say "Next: $bin_dir/brasskey doctor"
            ;;
    esac
}

main "$@"
