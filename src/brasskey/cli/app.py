# SPDX-License-Identifier: Apache-2.0
"""The ``brasskey`` command: Typer app, global options and the error boundary.

``main`` runs Click in non-standalone mode so this module, not Click, decides how every outcome is
rendered and which exit code it gets (see docs/decisions/0001):

- a command returns a :class:`CommandResult`, rendered here with its ``Next:`` footer;
- a :class:`BrasskeyError` becomes a plain message with a fix;
- a usage error exits 2 with a pointer to ``--help``;
- Ctrl-C exits 130 (Typer converts KeyboardInterrupt into ``Exit(130)``);
- anything unexpected gets a short message, and a traceback only with ``--debug``.
"""

import logging
import os
import platform
import shlex
import sys
from collections.abc import Sequence
from typing import Annotated

import typer
from pydantic import JsonValue
from rich.text import Text

from brasskey import __version__
from brasskey.cli.bench import bench_command
from brasskey.cli.doctor import doctor_command
from brasskey.cli.options import DebugFlag, JsonFlag, YesFlag, apply_flags
from brasskey.cli.run import run_command
from brasskey.errors import ISSUES_URL, BrasskeyError, ExitCode
from brasskey.log import configure_logging
from brasskey.output.next_step import SEE_COMMANDS, NextStep
from brasskey.output.render import emit_brasskey_error, emit_error, emit_result, emit_traceback
from brasskey.output.result import CommandResult
from brasskey.output.settings import OutputSettings, use_settings

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="brasskey",
    help=(
        "Brasskey: the key to every engine.\n\n"
        "Deploy, tune, verify and serve open models on any hardware, "
        "measured on your own traffic."
    ),
    add_completion=False,
    pretty_exceptions_enable=False,  # main() owns error rendering
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command(name="run")(run_command)
app.command(name="bench")(bench_command)
app.command(name="doctor")(doctor_command)


def _platform_label() -> str:
    """Human-friendly OS label, e.g. ``macOS 15.4 (arm64)``."""
    machine = platform.machine() or "unknown"
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]} ({machine})"
    return f"{platform.system()} {platform.release()} ({machine})"


def version_result() -> CommandResult:
    """Version details, useful in bug reports."""
    data: dict[str, JsonValue] = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": _platform_label(),
    }
    human = Text(f"brasskey {__version__}\nPython {data['python']} on {data['platform']}")
    return CommandResult(command="version", data=data, next_step=SEE_COMMANDS, human=human)


def _command_names() -> list[str]:
    """Registered top-level command names, sorted."""
    return sorted(info.name for info in app.registered_commands if info.name)


def _overview_result() -> CommandResult:
    """``brasskey --json`` with no command: list commands for machine discovery."""
    names: list[JsonValue] = list(_command_names())
    return CommandResult(
        command="brasskey",
        data={"version": __version__, "commands": names},
        next_step=SEE_COMMANDS,
    )


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Show the Brasskey version and exit.")
    ] = False,
    json_output: JsonFlag = False,
    debug: DebugFlag = False,
    yes: YesFlag = False,
) -> CommandResult | None:
    """Global options; runs before any command."""
    settings = apply_flags(
        ctx.invoked_subcommand or "brasskey", json_output=json_output, debug=debug, yes=yes
    )
    if version:
        # Handled here so `brasskey --version <command>` prints the version and stops.
        emit_result(version_result(), settings)
        raise typer.Exit(int(ExitCode.OK))
    if ctx.invoked_subcommand is None:
        if settings.json:
            return _overview_result()
        # Typer prints its rich help directly; echo only if it returned plain text.
        help_text = ctx.get_help()
        if help_text:
            typer.echo(help_text)
        raise typer.Exit(int(ExitCode.OK))
    return None


def _cli_error(error: typer.TyperException, settings: OutputSettings) -> int:
    """Invalid command line (or another Typer/Click failure): what is wrong and where to look.

    Typer 0.27+ vendors Click; its exceptions derive from the public ``typer.TyperException``.
    Usage errors carry the Click context, from which we take the command path for the hint.
    """
    context_path: object = getattr(getattr(error, "ctx", None), "command_path", None)
    command_path = context_path if isinstance(context_path, str) else "brasskey"
    emit_error(
        command=settings.command,
        code="BK-E-USAGE" if error.exit_code == ExitCode.USAGE else "BK-E-CLI",
        message=error.format_message(),
        fix=f"See valid usage with `{command_path} --help`.",
        next_step=NextStep(command=f"{command_path} --help", hint="see valid options"),
        settings=settings,
    )
    return error.exit_code


def _interrupted(settings: OutputSettings) -> int:
    emit_error(
        command=settings.command,
        code="BK-E-INTERRUPTED",
        message="Stopped (Ctrl-C).",
        fix=None,
        next_step=None,
        settings=settings,
    )
    return int(ExitCode.INTERRUPTED)


def _internal_error(error: Exception, args: Sequence[str], settings: OutputSettings) -> int:
    """A bug in Brasskey: short message by default, full traceback with --debug."""
    logger.debug("Unhandled %s in command %s", type(error).__name__, settings.command)
    if settings.debug:
        emit_traceback(error)
        fix = f"Please report this at {ISSUES_URL} and include the traceback above."
        next_step = None
    else:
        fix = f"Re-run with --debug to see the full traceback, then report it at {ISSUES_URL}."
        rerun = shlex.join(["brasskey", "--debug", *args])
        next_step = NextStep(command=rerun, hint="show the full traceback")
    emit_error(
        command=settings.command,
        code="BK-E-INTERNAL",
        message=f"Brasskey hit an unexpected error ({type(error).__name__}: {error}).",
        fix=fix,
        next_step=next_step,
        settings=settings,
    )
    return int(ExitCode.ERROR)


def _invoke(args: list[str], settings: OutputSettings) -> int:
    """Run the Click command and turn every outcome into rendered output and an exit code."""
    command = typer.main.get_command(app)
    try:
        outcome: object = command.main(args=args, prog_name="brasskey", standalone_mode=False)
    except typer.TyperException as error:  # usage errors and other Click failures
        return _cli_error(error, settings)
    except (typer.Abort, KeyboardInterrupt):  # EOF at a prompt, or Ctrl-C outside Typer's loop
        return _interrupted(settings)
    except BrasskeyError as error:
        logger.debug("BrasskeyError %s in command %s", error.code, settings.command)
        emit_brasskey_error(error, settings)
        return int(error.exit_code)
    except Exception as error:  # last-resort boundary: never show a raw traceback by default
        return _internal_error(error, args, settings)

    if isinstance(outcome, CommandResult):
        emit_result(outcome, settings)
        return int(outcome.exit_code)
    if outcome == ExitCode.INTERRUPTED:  # Typer turned Ctrl-C into Exit(130)
        return _interrupted(settings)
    if isinstance(outcome, int):  # exit code from --help or typer.Exit
        return outcome
    return int(ExitCode.OK)


def _discard_stdout() -> None:
    """Point stdout at /dev/null so Python's final flush doesn't report the broken pipe again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass  # stdout has no real file descriptor (e.g. captured in tests)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``brasskey`` console script and ``python -m brasskey``."""
    args = list(sys.argv[1:] if argv is None else argv)
    settings = OutputSettings.prescan(args)
    with use_settings(settings):
        configure_logging(debug=settings.debug, json_mode=settings.json)
        logger.debug(
            "brasskey %s starting (command=%s, json=%s)",
            __version__,
            settings.command,
            settings.json,
        )
        try:
            exit_code = _invoke(args, settings)
        except BrokenPipeError:
            # The reader stopped early (e.g. `brasskey doctor --json | head`): exit quietly.
            _discard_stdout()
            logger.debug("stdout closed by the reader; exiting quietly")
            return int(ExitCode.BROKEN_PIPE)
        logger.debug("brasskey exiting with code %d", exit_code)
        return exit_code
