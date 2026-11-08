# SPDX-License-Identifier: Apache-2.0
"""One-time setup, run after the user confirms: the pinned uv environment for MLX or GuideLLM,
or pulling the pinned vLLM / SGLang image."""

import logging
import os
import subprocess
from collections.abc import Sequence

from brasskey.engines.base import ENGINE_BASE_ENV, EngineInstall
from brasskey.errors import BrasskeyError

logger = logging.getLogger(__name__)

DEFAULT_SETUP_TIMEOUT_S = 1800.0


class EngineSetupError(BrasskeyError):
    default_code = "BK-E-ENGINE-SETUP"


def run_setup(
    argv: Sequence[str], *, name: str, timeout_s: float = DEFAULT_SETUP_TIMEOUT_S
) -> None:
    """Run a setup command (downloads and caches a tool). Raises a plain error."""
    logger.debug("Setting up %s: %s", name, argv[0])
    try:
        # argv is built by Brasskey from fixed arguments, never from user input.
        completed = subprocess.run(  # noqa: S603
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env={**os.environ, **ENGINE_BASE_ENV, "HF_HUB_OFFLINE": "0"},
        )
    except subprocess.TimeoutExpired as error:
        raise EngineSetupError(
            f"Setting up {name} took longer than {timeout_s / 60:.0f} minutes.",
            fix="Check your internet connection and run again.",
        ) from error
    except FileNotFoundError as error:
        raise EngineSetupError(
            f"Setting up {name} needs {argv[0]}, which wasn't found.",
            fix=f"Install {argv[0]} and run again (`brasskey doctor` shows what is installed).",
        ) from error
    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        last = lines[-1] if lines else "no output"
        raise EngineSetupError(
            f"Setting up {name} failed: {last}",
            fix="Check your internet connection and run again; add --debug for details.",
        )
    logger.debug("%s set up", name)


def run_engine_setup(install: EngineInstall, *, timeout_s: float | None = None) -> None:
    """Run ``install.setup_argv`` (downloads and caches the engine). Raises a plain error."""
    if not install.needs_setup or not install.setup_argv:
        return
    run_setup(
        install.setup_argv,
        name=install.engine.display_name,
        timeout_s=timeout_s if timeout_s is not None else install.setup_timeout_s,
    )
