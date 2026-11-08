# SPDX-License-Identifier: Apache-2.0
"""Keep a server running in the foreground until Ctrl-C (or SIGTERM), then stop it cleanly."""

import logging
import signal
import threading
from collections.abc import Callable
from types import FrameType

from brasskey.runtime.supervisor import ProcessSupervisor, ServerHandle

logger = logging.getLogger(__name__)


class StopRequestedError(Exception):
    """SIGTERM asked Brasskey to stop (handled like Ctrl-C)."""


def _raise_stop(_signum: int, _frame: FrameType | None) -> None:
    raise StopRequestedError


def serve_until_stopped(
    handle: ServerHandle,
    supervisor: ProcessSupervisor,
    *,
    wait: Callable[[], int] | None = None,
) -> int | None:
    """Block until the user stops Brasskey (returns None) or the engine exits (its exit code).

    The engine always gets stopped on the way out, whatever happens here.
    """
    waiter = wait if wait is not None else handle.process.wait
    on_main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _raise_stop) if on_main else None
    try:
        code = waiter()
        logger.debug("%s exited by itself with code %s", handle.engine_name, code)
        return code
    except (KeyboardInterrupt, StopRequestedError):
        logger.debug("Stop requested; stopping %s", handle.engine_name)
        supervisor.stop(handle)
        return None
    finally:
        if handle.process.poll() is None:
            supervisor.stop(handle)
        if on_main and previous is not None:
            signal.signal(signal.SIGTERM, previous)
