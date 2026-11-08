# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the tests."""

import secrets
import socket
import threading
from typing import IO


def fake_api_key(label: str = "test") -> str:
    """A strong, random Brasskey-format key for one test run.

    Generated at run time so no key-like string is ever committed: secret scanners stay quiet,
    and nothing in the repository could be mistaken for a real credential.
    """
    return f"bk-{label}-{secrets.token_urlsafe(24)}"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def read_line(stream: IO[str], timeout_s: float) -> str:
    """One line from a subprocess pipe, or TimeoutError."""
    box: list[str] = []
    reader = threading.Thread(target=lambda: box.append(stream.readline()), daemon=True)
    reader.start()
    reader.join(timeout_s)
    if not box:
        raise TimeoutError(f"no output within {timeout_s:.0f}s")
    return box[0]
