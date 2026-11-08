# SPDX-License-Identifier: Apache-2.0
"""Logging setup for the CLI.

Logs always go to stderr so stdout stays clean for results and ``--json``. The default level is
WARNING; ``--debug`` switches to DEBUG. Modules log through ``logging.getLogger(__name__)``, which
places them under the ``brasskey`` logger configured here.

Privacy (CONTRIBUTING.md): never log prompt or response content, API keys or tokens. Log
identifiers, sizes, timings and decisions instead. As a safety net, API keys registered with
:func:`register_secret` are replaced by ``[redacted]`` in every log line.
"""

import logging
import sys

from rich.logging import RichHandler

from brasskey.output.render import stderr_console

LOGGER_NAME = "brasskey"
REDACTED = "[redacted]"
_SECRETS: set[str] = set()


def register_secret(value: str) -> None:
    """Never let ``value`` (an API key) appear in any Brasskey log line."""
    if len(value) >= 8:  # too-short strings would redact ordinary words
        _SECRETS.add(value)


def redact(text: str) -> str:
    for secret in _SECRETS:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


class _RedactingFilter(logging.Filter):
    """Rewrites each record's final message with registered secrets removed."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _SECRETS:
            message = redact(record.getMessage())
            record.msg = message
            record.args = None
        return True


class _HandlerSlot:
    """Remembers the handler we installed so reconfiguring replaces it instead of stacking."""

    handler: logging.Handler | None = None


def configure_logging(*, debug: bool, json_mode: bool) -> None:
    """(Re)configure the ``brasskey`` logger for this invocation. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    if _HandlerSlot.handler is not None:
        logger.removeHandler(_HandlerSlot.handler)

    handler: logging.Handler
    if json_mode:
        # Plain, greppable lines next to machine output.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    else:
        handler = RichHandler(
            console=stderr_console(),
            show_time=debug,
            show_path=debug,
            markup=False,
            rich_tracebacks=False,
        )

    handler.addFilter(_RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    # The CLI owns output formatting; don't double-print through the root logger.
    logger.propagate = False
    _HandlerSlot.handler = handler
