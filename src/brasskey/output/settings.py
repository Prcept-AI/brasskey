# SPDX-License-Identifier: Apache-2.0
"""Per-invocation output settings (--json, --debug, --yes).

The flags can appear before the command (``brasskey --json run``) or after it
(``brasskey run --json``). ``prescan`` reads them straight from argv so that errors raised while
the command line is still being parsed (usage errors) are rendered in the requested mode; the
parsed options then update the same settings object.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

JSON_FLAG = "--json"
DEBUG_FLAG = "--debug"
YES_FLAGS = frozenset({"--yes", "-y"})


@dataclass(slots=True)
class OutputSettings:
    """How the current invocation should talk to the user."""

    json: bool = False
    debug: bool = False
    yes: bool = False
    command: str = "brasskey"

    @classmethod
    def prescan(cls, argv: Sequence[str]) -> "OutputSettings":
        """Best-effort read of the global flags from raw argv (stops at a ``--`` separator).

        Global options take no values today, so the first token that does not start with ``-``
        is the command name. Update this if a global option ever takes a value.
        """
        settings = cls()
        for arg in argv:
            if arg == "--":
                break
            if arg == JSON_FLAG:
                settings.json = True
            elif arg == DEBUG_FLAG:
                settings.debug = True
            elif arg in YES_FLAGS:
                settings.yes = True
            elif not arg.startswith("-") and settings.command == "brasskey":
                settings.command = arg
        return settings


_current: ContextVar[OutputSettings | None] = ContextVar("brasskey_output_settings", default=None)


def current_settings() -> OutputSettings:
    """Settings for the running invocation (defaults when called outside ``main``)."""
    settings = _current.get()
    if settings is None:
        settings = OutputSettings()
        _current.set(settings)
    return settings


@contextmanager
def use_settings(settings: OutputSettings) -> Generator[OutputSettings, None, None]:
    """Install ``settings`` for the duration of one CLI invocation."""
    token = _current.set(settings)
    try:
        yield settings
    finally:
        _current.reset(token)
