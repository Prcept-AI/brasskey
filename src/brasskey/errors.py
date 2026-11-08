# SPDX-License-Identifier: Apache-2.0
"""Errors and exit codes.

Every error a user can hit is a :class:`BrasskeyError` with a plain-language message and a concrete
fix, and full tracebacks only with ``--debug`` (CONTRIBUTING.md, CLI contract).
Exit codes are stable so scripts and CI can rely on them (see docs/decisions/0001).
"""

from enum import IntEnum

from brasskey.output.next_step import SEE_COMMANDS, NextStep

PROJECT_URL = "https://github.com/Prcept-AI/brasskey"
ISSUES_URL = f"{PROJECT_URL}/issues"


class ExitCode(IntEnum):
    """Process exit codes shared by every command."""

    OK = 0  # success; also a PASS verdict
    ERROR = 1  # something went wrong; the message says what and how to fix it
    USAGE = 2  # invalid command line (same as Click's default)
    CHECK_FAILED = 3  # a check or comparison ran and FAILED (doctor, bench --compare)
    INCONCLUSIVE = 4  # a comparison ran but the data cannot support a verdict
    INTERRUPTED = 130  # stopped with Ctrl-C (128 + SIGINT)
    BROKEN_PIPE = 141  # the reader of our output went away, e.g. `| head` (128 + SIGPIPE)


class BrasskeyError(Exception):
    """An expected failure with a plain-language explanation and a fix.

    ``code`` is a stable identifier (``BK-E-...``) for docs, search and ``--json`` consumers.
    """

    default_code: str = "BK-E-GENERIC"
    default_exit_code: ExitCode = ExitCode.ERROR

    def __init__(
        self,
        message: str,
        *,
        fix: str | None = None,
        code: str | None = None,
        exit_code: ExitCode | None = None,
        next_step: NextStep | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.code = code or self.default_code
        self.exit_code = exit_code if exit_code is not None else self.default_exit_code
        self.next_step = next_step if next_step is not None else SEE_COMMANDS


class NotAvailableYetError(BrasskeyError):
    """Raised by commands that are registered but not built yet in this pre-alpha."""

    default_code = "BK-E-NOT-AVAILABLE"

    def __init__(self, command: str) -> None:
        super().__init__(
            f"`{command}` isn't available yet in this pre-alpha build.",
            fix=f"Follow progress at {PROJECT_URL}.",
        )


class ConfirmationRequiredError(BrasskeyError):
    """A heavy action needs confirmation but nobody can answer a prompt (non-TTY or --json)."""

    default_code = "BK-E-CONFIRM"

    def __init__(self, question: str) -> None:
        super().__init__(
            f"Brasskey needs confirmation before continuing: {question}",
            fix="Re-run with --yes to proceed without a prompt.",
        )
