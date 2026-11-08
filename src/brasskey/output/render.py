# SPDX-License-Identifier: Apache-2.0
"""Render results and errors for people (Rich) or machines (--json).

Rules this module enforces (CONTRIBUTING.md, CLI contract):
- every command ends with a one-line ``Next:`` suggestion;
- with ``--json``, stdout carries exactly one JSON document; everything else goes to stderr;
- errors are plain language with a fix; tracebacks only with ``--debug``.

Text is assembled with :class:`rich.text.Text` rather than markup strings, so messages that contain
square brackets (paths, model names, user input) are printed literally.
"""

import sys
from types import TracebackType

from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

from brasskey.errors import BrasskeyError
from brasskey.output.envelope import Envelope, ErrorInfo
from brasskey.output.next_step import NextStep
from brasskey.output.result import CommandResult
from brasskey.output.settings import OutputSettings

# Brand accent (brand/BRAND.md): Patina. Used only for labels; message text keeps the default color.
NEXT_STYLE = "bold #2BB3A3"
ERROR_STYLE = "bold red"
FIX_STYLE = "bold"


def stdout_console() -> Console:
    """Console for results. soft_wrap keeps long commands on one line for copy and paste."""
    return Console(highlight=False, soft_wrap=True)


def stderr_console() -> Console:
    """Console for errors, logs, prompts and progress."""
    return Console(stderr=True, highlight=False, soft_wrap=True)


def next_line(step: NextStep) -> Text:
    """The ``Next:`` footer as styled text."""
    return Text.assemble(("Next: ", NEXT_STYLE), step.as_text())


def _write_json(envelope: Envelope) -> None:
    sys.stdout.write(envelope.to_json() + "\n")
    sys.stdout.flush()


def emit_result(result: CommandResult, settings: OutputSettings) -> None:
    """Print a finished command's output, ending with the ``Next:`` line."""
    if settings.json:
        _write_json(
            Envelope(command=result.command, ok=True, data=result.data, next=result.next_step)
        )
        return
    console = stdout_console()
    if result.human is not None:
        console.print(result.human)
        console.print()
    console.print(next_line(result.next_step))


def emit_error(
    *,
    command: str,
    code: str,
    message: str,
    fix: str | None,
    next_step: NextStep | None,
    settings: OutputSettings,
) -> None:
    """Print an error: JSON envelope on stdout with --json, plain text on stderr otherwise."""
    if settings.json:
        _write_json(
            Envelope(
                command=command,
                ok=False,
                error=ErrorInfo(code=code, message=message, fix=fix),
                next=next_step,
            )
        )
        return
    console = stderr_console()
    console.print(Text.assemble(("Error: ", ERROR_STYLE), message))
    if fix:
        console.print(Text.assemble(("Fix: ", FIX_STYLE), fix))
    if next_step is not None:
        console.print(next_line(next_step))


def emit_brasskey_error(error: BrasskeyError, settings: OutputSettings) -> None:
    """Print a :class:`BrasskeyError` with its code, fix and next step."""
    emit_error(
        command=settings.command,
        code=error.code,
        message=error.message,
        fix=error.fix,
        next_step=error.next_step,
        settings=settings,
    )


def emit_traceback(error: BaseException) -> None:
    """Print a full traceback to stderr (only called with --debug).

    Locals are not shown: they could hold API keys or other secrets.
    """
    trace: TracebackType | None = error.__traceback__
    stderr_console().print(Traceback.from_exception(type(error), error, trace, show_locals=False))
