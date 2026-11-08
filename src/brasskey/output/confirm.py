# SPDX-License-Identifier: Apache-2.0
"""Confirmation before heavy actions (downloads, GPU memory use, long benchmarks).

``--yes`` skips the prompt. When nobody can answer (stdin is not a terminal, or ``--json`` is on)
Brasskey stops with a clear error instead of hanging or silently proceeding.
"""

import logging
import sys
from collections.abc import Callable

from rich.prompt import Confirm

from brasskey.errors import ConfirmationRequiredError
from brasskey.output.render import stderr_console
from brasskey.output.settings import OutputSettings, current_settings

logger = logging.getLogger(__name__)


def _stdin_is_tty() -> bool:
    stdin = sys.stdin
    return stdin is not None and stdin.isatty()


def confirm(
    question: str,
    *,
    settings: OutputSettings | None = None,
    default: bool = False,
    is_interactive: Callable[[], bool] = _stdin_is_tty,
) -> bool:
    """Ask a yes/no question on stderr and return the answer.

    Raises :class:`ConfirmationRequiredError` when no one can answer and ``--yes`` was not given.
    """
    active = settings if settings is not None else current_settings()
    if active.yes:
        logger.debug("Confirmation skipped by --yes: %s", question)
        return True
    if active.json or not is_interactive():
        logger.debug("Cannot prompt (json=%s, interactive=False): %s", active.json, question)
        raise ConfirmationRequiredError(question)
    answer = Confirm.ask(question, default=default, console=stderr_console())
    logger.debug("Confirmation answer=%s: %s", answer, question)
    return answer
