# SPDX-License-Identifier: Apache-2.0
"""Options every command accepts: --json, --debug and --yes.

They work before the command (``brasskey --json run``) and after it (``brasskey run --json``).
Each command calls :func:`apply_flags` first so both positions end up in the same settings.
"""

import logging
from typing import Annotated

import typer

from brasskey.log import configure_logging
from brasskey.output.settings import OutputSettings, current_settings

logger = logging.getLogger(__name__)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", help="Print one JSON document to stdout instead of text."),
]
DebugFlag = Annotated[
    bool,
    typer.Option("--debug", help="Show debug logs and full error tracebacks on stderr."),
]
YesFlag = Annotated[
    bool,
    typer.Option("--yes", "-y", help="Answer yes to confirmation prompts."),
]


def apply_flags(command: str, *, json_output: bool, debug: bool, yes: bool) -> OutputSettings:
    """Merge a command's flags into the invocation settings and refresh logging if needed."""
    settings = current_settings()
    settings.command = command
    logging_changed = (json_output and not settings.json) or (debug and not settings.debug)
    settings.json = settings.json or json_output
    settings.debug = settings.debug or debug
    settings.yes = settings.yes or yes
    if logging_changed:
        configure_logging(debug=settings.debug, json_mode=settings.json)
    logger.debug(
        "Flags for %s: json=%s debug=%s yes=%s",
        command,
        settings.json,
        settings.debug,
        settings.yes,
    )
    return settings
