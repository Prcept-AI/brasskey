# SPDX-License-Identifier: Apache-2.0
"""Read-only facts about this machine's network exposure, for ``doctor --security``.

Everything here observes; nothing changes settings, and nothing contacts other machines. The only
network traffic is to this machine's own listening ports (and to a URL the user names).
"""

import ipaddress
import json
import logging
import re
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import httpx2
import psutil

from brasskey.hardware.commands import CommandRunner, run_command

logger = logging.getLogger(__name__)

LSOF = "/usr/sbin/lsof"
SOCKETFILTERFW = "/usr/libexec/ApplicationFirewall/socketfilterfw"
PROBE_ORIGIN = "http://brasskey-origin-check.invalid"  # a web page that should never be trusted
PROBE_TIMEOUT_S = 2.0
WILDCARD_ADDRESSES = frozenset({"*", "0.0.0.0", "::", "[::]"})  # noqa: S104


@dataclass(frozen=True)
class Listener:
    address: str
    port: int
    pid: int | None = None
    process: str | None = None

    @property
    def wildcard(self) -> bool:
        return self.address in WILDCARD_ADDRESSES

    @property
    def loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.address.strip("[]")).is_loopback
        except ValueError:
            return self.address == "localhost"


@dataclass(frozen=True)
class DockerPublish:
    container: str
    image: str
    host_address: str
    host_port: int
    container_port: int


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    cmdline: tuple[str, ...]


@dataclass(frozen=True)
class HttpProbe:
    """What unauthenticated requests get back.

    ``status`` and ``looks_like_models`` come from a plain request, as a script or another
    machine sends it: they decide whether a key is needed. ``allow_origin`` and ``origin_status``
    come from a request with a foreign ``Origin`` header, as a web page sends it: they decide
    whether any website could read the answers. One request can't answer both, because some
    servers refuse foreign origins (HTTP 403) while needing no key at all (Ollama, Brasskey).
    """

    url: str
    status: int
    allow_origin: str | None
    looks_like_models: bool
    origin_status: int | None = None  # with the foreign Origin; None: not sent (same as status)

    @property
    def open_without_key(self) -> bool:
        return self.status == 200 and self.looks_like_models

    @property
    def readable_by_any_site(self) -> bool:
        status = self.origin_status if self.origin_status is not None else self.status
        return status == 200 and self.allow_origin in ("*", PROBE_ORIGIN)


def _split_address(name: str) -> tuple[str, int] | None:
    """lsof NAME such as ``*:11434``, ``127.0.0.1:8000`` or ``[::1]:8000``."""
    address, _, port = name.rpartition(":")
    return (address, int(port)) if port.isdigit() and address else None


def parse_lsof(text: str) -> list[Listener]:
    """``lsof -nP -iTCP -sTCP:LISTEN -F pcn`` output (one field per line)."""
    listeners: list[Listener] = []
    pid: int | None = None
    command: str | None = None
    for line in text.splitlines():
        tag, value = line[:1], line[1:]
        if tag == "p":
            pid = int(value) if value.isdigit() else None
        elif tag == "c":
            command = value
        elif tag == "n":
            split = _split_address(value.split(" ")[0])
            if split is not None:
                listeners.append(Listener(split[0], split[1], pid, command))
    return listeners


def listening_sockets(run: CommandRunner = run_command) -> list[Listener]:
    """TCP sockets listening on this machine (other users' processes may lack a pid/name)."""
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, PermissionError):
        # macOS without root: lsof lists this user's own listeners.
        out = run([LSOF, "-nP", "-iTCP", "-sTCP:LISTEN", "-F", "pcn"], timeout_s=10)
        return sorted(set(parse_lsof(out.stdout)), key=lambda item: (item.port, item.address))
    found: set[Listener] = set()
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or not conn.laddr:
            continue
        name = None
        if conn.pid:
            try:
                name = psutil.Process(conn.pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = None
        found.add(Listener(str(conn.laddr.ip), int(conn.laddr.port), conn.pid, name))
    return sorted(found, key=lambda item: (item.port, item.address))


def interface_addresses() -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for entries in psutil.net_if_addrs().values():
        for entry in entries:
            if entry.family not in (socket.AF_INET, socket.AF_INET6):
                continue
            try:
                addresses.append(ipaddress.ip_address(entry.address.split("%")[0]))
            except ValueError:
                continue
    return addresses


_DOCKER_PORT = re.compile(r"^(?P<host>[0-9a-fA-F.:\[\]]*):(?P<hport>\d+)->(?P<cport>\d+)/tcp$")


def parse_docker_ps(text: str) -> list[DockerPublish]:
    """``docker ps --format '{{json .}}'`` lines."""
    published: list[DockerPublish] = []
    for line in text.splitlines():
        try:
            row: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        fields = cast(dict[str, object], row)
        for mapping in str(fields.get("Ports", "")).split(","):
            match = _DOCKER_PORT.match(mapping.strip())
            if match:
                published.append(
                    DockerPublish(
                        container=str(fields.get("Names", "")),
                        image=str(fields.get("Image", "")),
                        host_address=match.group("host"),
                        host_port=int(match.group("hport")),
                        container_port=int(match.group("cport")),
                    )
                )
    return published


def docker_publishes(run: CommandRunner = run_command) -> list[DockerPublish] | None:
    """Ports published by running containers; None when Docker isn't available."""
    out = run(["docker", "ps", "--format", "{{json .}}"], timeout_s=10)
    if not out.ok:
        return None
    return parse_docker_ps(out.stdout)


def firewall_state(
    run: CommandRunner = run_command, platform_name: str | None = None
) -> str | None:
    """ "on"/"off" for the macOS application firewall; None where it can't be read without root."""
    if (platform_name or sys.platform) != "darwin":
        return None
    out = run([SOCKETFILTERFW, "--getglobalstate"], timeout_s=5)
    text = out.stdout.lower()
    if "enabled" in text:
        return "on"
    if "disabled" in text:
        return "off"
    return None


def own_processes() -> list[ProcessInfo]:
    """This user's processes and command lines (to spot API keys passed as arguments)."""
    found: list[ProcessInfo] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = tuple(process.info.get("cmdline") or ())
            found.append(ProcessInfo(process.info["pid"], process.info.get("name") or "", cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


Prober = Callable[[str], HttpProbe | None]


def _looks_like_models(response: httpx2.Response) -> bool:
    """A 200 with an OpenAI-style (``data``) or Ollama-style (``models``) model list."""
    if response.status_code != 200:
        return False
    try:
        body: object = response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    fields = cast(dict[str, object], body)
    return isinstance(fields.get("data"), list) or isinstance(fields.get("models"), list)


def make_prober(client: httpx2.Client | None = None) -> Prober:
    """GET a models endpoint without a key: as a script would, then as a foreign web page would.

    See ``HttpProbe`` for why these are two requests.
    """
    active = client or httpx2.Client(timeout=PROBE_TIMEOUT_S, trust_env=False, verify=True)

    def probe(url: str) -> HttpProbe | None:
        try:
            plain = active.get(url)
        except httpx2.HTTPError:
            return None
        looks = _looks_like_models(plain)
        origin_status: int | None = None
        allow_origin: str | None = None
        if looks:  # only a server that answers without a key can leak answers to web pages
            try:
                foreign = active.get(url, headers={"Origin": PROBE_ORIGIN})
                origin_status = foreign.status_code
                allow_origin = foreign.headers.get("access-control-allow-origin")
            except httpx2.HTTPError:
                origin_status = 0  # no answer for a web page: nothing it could read
        logger.debug(
            "Probe %s: status=%d models=%s origin_status=%s allow_origin=%s",
            url,
            plain.status_code,
            looks,
            origin_status,
            allow_origin,
        )
        return HttpProbe(
            url=url,
            status=plain.status_code,
            allow_origin=allow_origin,
            looks_like_models=looks,
            origin_status=origin_status,
        )

    return probe
