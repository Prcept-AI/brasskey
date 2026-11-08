# SPDX-License-Identifier: Apache-2.0
"""Operating system detection (read-only)."""

import platform
import sys
from pathlib import Path

from brasskey.hardware.types import OsInfo


def _os_release_pretty_name(etc_root: Path) -> str | None:
    """PRETTY_NAME from /etc/os-release, e.g. "Ubuntu 24.04.1 LTS"."""
    try:
        text = (etc_root / "os-release").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "PRETTY_NAME":
            return value.strip().strip('"').strip("'") or None
    return None


def detect_os(
    *,
    platform_name: str | None = None,
    release: str | None = None,
    mac_version: str | None = None,
    etc_root: Path = Path("/etc"),
) -> OsInfo:
    """Describe the running OS. Arguments exist so tests can simulate other systems."""
    name = platform_name or sys.platform
    kernel = release if release is not None else platform.release()
    if name == "darwin":
        version = mac_version if mac_version is not None else platform.mac_ver()[0]
        version = version or kernel
        return OsInfo(family="darwin", version=version, description=f"macOS {version}")
    if name.startswith("linux"):
        pretty = _os_release_pretty_name(etc_root) or "Linux"
        description = f"{pretty} (kernel {kernel})"
        if "microsoft" in kernel.lower():
            description += " on WSL2"
        return OsInfo(family="linux", version=kernel, description=description)
    if name == "win32":
        return OsInfo(family="windows", version=kernel, description=f"Windows {kernel}")
    return OsInfo(family="other", version=kernel, description=f"{name} {kernel}".strip())
