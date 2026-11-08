# SPDX-License-Identifier: Apache-2.0
"""Run read-only system tools (sysctl, ioreg) for the probes.

Tools are called by absolute path, so a PATH entry can't substitute them, and with ``LC_ALL=C``
for stable output. The runner never raises; failures come back as a non-zero return code so a
probe can turn them into a plain-language issue. Tests inject a fake runner.
"""

import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

SYSCTL = "/usr/sbin/sysctl"
IOREG = "/usr/sbin/ioreg"


@dataclass(frozen=True, slots=True)
class CommandOutput:
    returncode: int
    stdout: str
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], *, timeout_s: float = 5.0) -> CommandOutput: ...


def run_command(argv: Sequence[str], *, timeout_s: float = 5.0) -> CommandOutput:
    """Run a system tool and capture its output. Never raises."""
    logger.debug("Running %s", argv[0])
    try:
        # argv is a fixed absolute path to an OS tool plus fixed arguments, never user input.
        completed = subprocess.run(  # noqa: S603
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except FileNotFoundError:
        return CommandOutput(127, "", f"{argv[0]} not found")
    except subprocess.TimeoutExpired:
        return CommandOutput(124, "", f"{argv[0]} timed out after {timeout_s:g}s")
    except OSError as error:
        return CommandOutput(126, "", str(error))
    return CommandOutput(completed.returncode, completed.stdout, completed.stderr)


def read_sysctl(run: CommandRunner, names: Sequence[str]) -> dict[str, str]:
    """Read sysctl values as ``{name: value}``. Unknown names are simply absent.

    One call for all names: sysctl prints ``name: value`` for each known name and complains on
    stderr about unknown ones (exiting non-zero), so stdout is parsed regardless of the code.
    """
    output = run([SYSCTL, *names])
    values: dict[str, str] = {}
    for line in output.stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            values[key.strip()] = value.strip()
    return values


def parse_int(value: str | None) -> int | None:
    """Parse an integer sysctl/sysfs value; None when missing or malformed."""
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None
