# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the real-engine integration tests."""

import socket
import threading
from typing import IO


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
