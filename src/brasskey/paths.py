# SPDX-License-Identifier: Apache-2.0
"""Where Brasskey keeps its own files (not model files, which live in the Hugging Face cache).

Defaults follow each OS's conventions (platformdirs). ``BRASSKEY_HOME`` puts everything under one
folder instead, which tests use so they never touch a real user's directories.

Directories Brasskey creates are private to the user (mode 0700): logs can include file paths
and state files describe running servers.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP_NAME = "brasskey"
HOME_ENV = "BRASSKEY_HOME"


@dataclass(frozen=True, slots=True)
class BrasskeyPaths:
    config_dir: Path
    state_dir: Path
    log_dir: Path

    @property
    def servers_dir(self) -> Path:
        """One JSON file per running server (see brasskey.runtime.state)."""
        return self.state_dir / "servers"


def resolve_paths(env: Mapping[str, str] | None = None) -> BrasskeyPaths:
    source = os.environ if env is None else env
    home = source.get(HOME_ENV)
    if home:
        base = Path(home).expanduser()
        return BrasskeyPaths(base / "config", base / "state", base / "logs")
    return BrasskeyPaths(
        config_dir=platformdirs.user_config_path(APP_NAME, appauthor=False),
        state_dir=platformdirs.user_state_path(APP_NAME, appauthor=False),
        log_dir=platformdirs.user_log_path(APP_NAME, appauthor=False),
    )


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) readable only by this user."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def open_private_file(path: Path, *, append: bool = False) -> int:
    """Open (create) a file with mode 0600 and return its file descriptor."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    return os.open(path, flags, 0o600)
