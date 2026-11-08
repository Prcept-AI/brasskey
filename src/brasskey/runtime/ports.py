# SPDX-License-Identifier: Apache-2.0
"""Choose a free TCP port for a server."""

import logging
import socket

from brasskey.errors import BrasskeyError

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8000
SEARCH_SPAN = 100


class PortInUseError(BrasskeyError):
    default_code = "BK-E-PORT-IN-USE"

    def __init__(self, message: str, *, fix: str | None = None, free_port: int | None = None):
        super().__init__(message, fix=fix)
        self.free_port = free_port  # free right now: the CLI suggests the same command with it


def _first_free(host: str) -> int | None:
    for port in range(DEFAULT_PORT, DEFAULT_PORT + SEARCH_SPAN):
        if port_is_free(host, port):
            return port
    return None


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def port_is_free(host: str, port: int) -> bool:
    """True if nothing listens on ``host:port`` and we could bind it.

    Both checks matter: on macOS a specific address can sometimes be bound while another program
    listens on the wildcard address for the same port.
    """
    family = _family(host)
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((host, port)) == 0:
            return False
    with socket.socket(family, socket.SOCK_STREAM) as binder:
        try:
            binder.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, requested: int | None) -> tuple[int, str | None]:
    """A free port and an optional note. A requested port must be free (no silent change)."""
    if requested is not None:
        if not port_is_free(host, requested):
            free = _first_free(host)
            logger.debug("Port %d on %s is busy; %s is free", requested, host, free)
            raise PortInUseError(
                f"Port {requested} on {host} is already in use.",
                fix=f"Pick another one with --port{f' ({free} is free)' if free else ''}, or stop "
                "the program using it.",
                free_port=free,
            )
        return requested, None
    port = _first_free(host)
    if port is not None:
        note = None if port == DEFAULT_PORT else f"Port {DEFAULT_PORT} is busy; using {port}."
        logger.debug("Chose port %d on %s", port, host)
        return port, note
    raise PortInUseError(
        f"No free port between {DEFAULT_PORT} and {DEFAULT_PORT + SEARCH_SPAN - 1} on {host}.",
        fix="Pass a free port with --port.",
    )
