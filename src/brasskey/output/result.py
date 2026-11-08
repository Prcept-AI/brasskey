# SPDX-License-Identifier: Apache-2.0
"""What a command returns: machine data, an optional human view, and the next step."""

from dataclasses import dataclass

from pydantic import JsonValue
from rich.console import RenderableType

from brasskey.errors import ExitCode
from brasskey.output.next_step import NextStep


@dataclass(frozen=True, slots=True)
class CommandResult:
    """A finished command, rendered by :func:`brasskey.output.render.emit_result`.

    ``data`` is the ``--json`` payload; ``human`` is the Rich view for terminals. Keeping both on
    one object makes it hard for the two output modes to drift apart.
    """

    command: str
    data: dict[str, JsonValue]
    next_step: NextStep
    human: RenderableType | None = None
    exit_code: ExitCode = ExitCode.OK
