# SPDX-License-Identifier: Apache-2.0
"""``brasskey doctor --security``: is any model server on this machine exposed? (ADR 0009)

The checks are pure functions over a :class:`SecurityContext`, gathered read-only by
:func:`gather_context`, so they are easy to test and to reuse from a future API. Each result has
a stable id (``BK-SEC-0NN``), a plain-language finding and, when something is wrong, a concrete
fix. Any FAIL makes the command exit with code 3.
"""

import ipaddress
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from brasskey.config.intent import is_loopback
from brasskey.paths import resolve_paths
from brasskey.runtime.state import ServerRecord, list_records
from brasskey.security.apikeys import ApiKey, default_keyring_get, key_weakness, resolve_api_key
from brasskey.security.netinfo import (
    DockerPublish,
    HttpProbe,
    Listener,
    ProcessInfo,
    docker_publishes,
    firewall_state,
    interface_addresses,
    listening_sockets,
    make_prober,
    own_processes,
)

logger = logging.getLogger(__name__)

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# Ports and process names of common local model servers.
KNOWN_PORTS = {
    11434: "Ollama",
    1234: "LM Studio",
    1337: "Jan",
    4891: "GPT4All",
    5000: "text-generation-webui",
    8000: "an OpenAI-compatible server (vLLM?)",
    8080: "llama.cpp or mlx-lm",
    9931: "llama.cpp",
    30000: "SGLang",
}
PROCESS_LABELS = (
    ("ollama", "Ollama"),
    ("llama-server", "llama.cpp"),
    ("llamafile", "llamafile"),
    ("mlx_lm", "mlx-lm"),
    ("vllm", "vLLM"),
    ("sglang", "SGLang"),
    ("lm studio", "LM Studio"),
    ("koboldcpp", "KoboldCpp"),
    ("local-ai", "LocalAI"),
)
MODEL_IMAGE_HINTS = (
    "vllm",
    "sglang",
    "ollama",
    "llama.cpp",
    "llama-cpp",
    "text-generation",
    "tgi",
    "lmdeploy",
    "localai",
    "tabbyapi",
)


class CheckStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - a check verdict, not a password
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    status: CheckStatus
    detail: str
    fix: str | None = None


@dataclass(frozen=True)
class SecurityContext:
    servers: Sequence[ServerRecord]
    listeners: Sequence[Listener]
    addresses: Sequence[IpAddress]
    docker: Sequence[DockerPublish] | None  # None: Docker isn't available
    firewall: str | None  # "on" / "off" / None (can't tell)
    processes: Sequence[ProcessInfo]
    api_key: ApiKey | None
    probe: Callable[[str], HttpProbe | None]
    os_family: str


@dataclass(frozen=True)
class Candidate:
    listener: Listener
    label: str
    evidence: str  # "brasskey" (state file), "process" (its name) or "port" (number only)

    @property
    def confirmed(self) -> bool:
        """Known to be a model server before probing (not just a familiar port number)."""
        return self.evidence != "port"


def _label(listener: Listener, brasskey_ports: set[int]) -> tuple[str, str] | None:
    if listener.port in brasskey_ports:
        return "Brasskey", "brasskey"
    name = (listener.process or "").lower()
    for hint, label in PROCESS_LABELS:
        if hint in name:
            return label, "process"
    known = KNOWN_PORTS.get(listener.port)
    return (known, "port") if known else None


def model_server_candidates(ctx: SecurityContext) -> list[Candidate]:
    """Listening sockets that look like model servers, widest exposure per port first."""
    brasskey_ports = {record.port for record in ctx.servers}
    engine_ports = {record.engine_port for record in ctx.servers if record.engine_port}
    by_port: dict[int, Candidate] = {}
    for listener in ctx.listeners:
        if listener.port in engine_ports:
            continue  # private loopback ports behind Brasskey's proxy
        labelled = _label(listener, brasskey_ports)
        if labelled is None:
            continue
        candidate = Candidate(listener, labelled[0], labelled[1])
        current = by_port.get(listener.port)
        if current is None or (not current.listener.wildcard and listener.wildcard):
            by_port[listener.port] = candidate
    return [by_port[port] for port in sorted(by_port)]


def probe_url(listener: Listener, label: str) -> str:
    address = listener.address.strip("[]")
    if listener.wildcard:
        host = "[::1]" if ":" in listener.address and listener.address != "*" else "127.0.0.1"
    else:
        host = f"[{address}]" if ":" in address else address
    path = "/api/tags" if label == "Ollama" else "/v1/models"
    return f"http://{host}:{listener.port}{path}"


def _exposed(listener: Listener) -> bool:
    return listener.wildcard or not listener.loopback


def _server_fix(label: str) -> str:
    if label == "Ollama":
        return (
            "Set OLLAMA_HOST=127.0.0.1:11434 for Ollama and restart it (or put it behind an "
            "authenticating proxy)."
        )
    return "Bind it to 127.0.0.1, or require an API key (`brasskey run` does both)."


def check_brasskey_servers(ctx: SecurityContext) -> list[CheckResult]:
    if not ctx.servers:
        return [
            CheckResult(
                id="BK-SEC-001",
                title="Brasskey servers",
                status=CheckStatus.SKIP,
                detail="No Brasskey servers are running.",
            )
        ]
    results: list[CheckResult] = []
    for record in ctx.servers:
        where = f"{record.model_id} on {record.host}:{record.port}"
        if is_loopback(record.host):
            status, detail, fix = CheckStatus.PASS, f"{where} listens on this machine only.", None
        elif record.auth != "api_key":
            status, detail = CheckStatus.FAIL, f"{where} accepts requests without an API key."
            fix = "Stop it and start it again with `brasskey run` (it requires a key off-loopback)."
        elif not record.tls:
            status = CheckStatus.WARN
            detail = f"{where} requires an API key, but traffic is not encrypted."
            fix = "Restart with --tls-cert/--tls-key, or put it behind a TLS reverse proxy."
        else:
            status, detail, fix = CheckStatus.PASS, f"{where} requires an API key over HTTPS.", None
        results.append(
            CheckResult(
                id="BK-SEC-001", title="Brasskey servers", status=status, detail=detail, fix=fix
            )
        )
    return results


def check_model_servers(
    ctx: SecurityContext, probes: dict[int, HttpProbe | None]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for candidate in model_server_candidates(ctx):
        found = probes.get(candidate.listener.port)
        if found is None:
            continue  # not answering HTTP like a model server
        if not candidate.confirmed and not found.looks_like_models:
            continue  # a familiar port number, but some other service (e.g. AirPlay on 5000)
        where = f"{candidate.label} on port {candidate.listener.port}"
        if not found.open_without_key:
            results.append(
                CheckResult(
                    id="BK-SEC-002",
                    title="Model servers need a key",
                    status=CheckStatus.PASS,
                    detail=f"{where} asks for an API key.",
                )
            )
        elif _exposed(candidate.listener):
            results.append(
                CheckResult(
                    id="BK-SEC-002",
                    title="Model servers need a key",
                    status=CheckStatus.FAIL,
                    detail=f"{where} answers anyone who can reach it on the network, without a "
                    f"key (it listens on {candidate.listener.address}).",
                    fix=_server_fix(candidate.label),
                )
            )
        else:
            results.append(
                CheckResult(
                    id="BK-SEC-002",
                    title="Model servers need a key",
                    status=CheckStatus.PASS,
                    detail=f"{where} needs no key but listens on this machine only.",
                )
            )
    if not results:
        results.append(
            CheckResult(
                id="BK-SEC-002",
                title="Model servers need a key",
                status=CheckStatus.SKIP,
                detail="No model servers found listening.",
            )
        )
    return results


def check_browser_access(
    ctx: SecurityContext, probes: dict[int, HttpProbe | None]
) -> list[CheckResult]:
    flagged: list[CheckResult] = []
    probed = 0
    for candidate in model_server_candidates(ctx):
        found = probes.get(candidate.listener.port)
        if found is None or (not candidate.confirmed and not found.looks_like_models):
            continue
        probed += 1
        if found.open_without_key and found.readable_by_any_site:
            flagged.append(
                CheckResult(
                    id="BK-SEC-003",
                    title="Web pages can't use model servers",
                    status=CheckStatus.WARN,
                    detail=f"Any website you open could use {candidate.label} on port "
                    f"{candidate.listener.port} and read its answers (it allows every origin).",
                    fix="Restrict its allowed origins (mlx-lm: --allowed-origins; Ollama: "
                    "OLLAMA_ORIGINS), or serve the model with `brasskey run`.",
                )
            )
    if flagged:
        return flagged
    status = CheckStatus.PASS if probed else CheckStatus.SKIP
    detail = (
        "No model server lets other websites read its answers."
        if probed
        else "No model servers found listening."
    )
    return [
        CheckResult(
            id="BK-SEC-003", title="Web pages can't use model servers", status=status, detail=detail
        )
    ]


def check_public_address(ctx: SecurityContext) -> list[CheckResult]:
    public = [address for address in ctx.addresses if address.is_global]
    wide = [c for c in model_server_candidates(ctx) if c.listener.wildcard]
    title = "Public network address"
    if not public:
        return [
            CheckResult(
                id="BK-SEC-004",
                title=title,
                status=CheckStatus.PASS,
                detail="This machine has no public IP address on its network interfaces.",
            )
        ]
    shown = ", ".join(str(address) for address in public[:2])
    if wide:
        ports = ", ".join(str(c.listener.port) for c in wide)
        return [
            CheckResult(
                id="BK-SEC-004",
                title=title,
                status=CheckStatus.FAIL,
                detail=f"This machine has a public IP address ({shown}) and model servers listen "
                f"on all interfaces (ports {ports}): they may be reachable from the internet.",
                fix="Bind them to 127.0.0.1, or make sure a firewall blocks those ports.",
            )
        ]
    return [
        CheckResult(
            id="BK-SEC-004",
            title=title,
            status=CheckStatus.PASS,
            detail=f"This machine has a public IP address ({shown}), but no model "
            "server listens on all interfaces.",
        )
    ]


def check_docker(ctx: SecurityContext) -> list[CheckResult]:
    title = "Docker port publishing"
    if ctx.docker is None:
        return [
            CheckResult(
                id="BK-SEC-005",
                title=title,
                status=CheckStatus.SKIP,
                detail="Docker isn't running (or isn't installed).",
            )
        ]
    risky = [
        item
        for item in ctx.docker
        if item.host_address in ("0.0.0.0", "::", "[::]", "")  # noqa: S104
        and (
            any(hint in item.image.lower() for hint in MODEL_IMAGE_HINTS)
            or item.host_port in KNOWN_PORTS
        )
    ]
    if not risky:
        return [
            CheckResult(
                id="BK-SEC-005",
                title=title,
                status=CheckStatus.PASS,
                detail="No model containers publish ports on all interfaces.",
            )
        ]
    return [
        CheckResult(
            id="BK-SEC-005",
            title=title,
            status=CheckStatus.WARN,
            detail=f"Container {item.container} ({item.image}) publishes port {item.host_port} "
            "on all network interfaces.",
            fix=f"Publish it on loopback only: -p 127.0.0.1:{item.host_port}:{item.container_port}"
            " (on Linux, Docker's published ports bypass ufw).",
        )
        for item in risky
    ]


def check_firewall(ctx: SecurityContext) -> list[CheckResult]:
    title = "Firewall"
    if ctx.firewall == "on":
        return [
            CheckResult(
                id="BK-SEC-006",
                title=title,
                status=CheckStatus.PASS,
                detail="The macOS firewall is on.",
            )
        ]
    if ctx.firewall == "off":
        return [
            CheckResult(
                id="BK-SEC-006",
                title=title,
                status=CheckStatus.WARN,
                detail="The macOS firewall is off.",
                fix="Turn it on in System Settings > Network > Firewall.",
            )
        ]
    hint = "check it with `sudo ufw status`" if ctx.os_family == "linux" else "check it manually"
    return [
        CheckResult(
            id="BK-SEC-006",
            title=title,
            status=CheckStatus.SKIP,
            detail=f"Reading the firewall needs administrator rights; {hint}.",
        )
    ]


SECRET_NAME_PARTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD")
ENV_FLAGS = ("-e", "--env")


def _secret_assignment(text: str) -> bool:
    """``NAME=value`` where NAME looks like a credential (``-e NAME`` alone passes it by name)."""
    name, separator, value = text.partition("=")
    return bool(separator and value) and any(part in name.upper() for part in SECRET_NAME_PARTS)


def _key_on_command_line(cmdline: Sequence[str]) -> bool:
    for index, arg in enumerate(cmdline):
        following = cmdline[index + 1] if index + 1 < len(cmdline) else None
        if arg.startswith("--api-key="):
            return True
        if arg == "--api-key" and following is not None:
            return True
        # Docker and similar tools: `-e VLLM_API_KEY=sk-...`, `--env=HF_TOKEN=...`.
        if arg in ENV_FLAGS and following is not None and _secret_assignment(following):
            return True
        if arg.startswith("--env=") and _secret_assignment(arg.removeprefix("--env=")):
            return True
    return False


def check_keys_on_command_lines(ctx: SecurityContext) -> list[CheckResult]:
    title = "API keys on command lines"
    exposed = [process for process in ctx.processes if _key_on_command_line(process.cmdline)]
    if not exposed:
        return [
            CheckResult(
                id="BK-SEC-007",
                title=title,
                status=CheckStatus.PASS,
                detail="No running program has an API key in its command line.",
            )
        ]
    return [
        CheckResult(
            id="BK-SEC-007",
            title=title,
            status=CheckStatus.WARN,
            detail=f"{process.name} (pid {process.pid}) has an API key in its command line, which "
            "other users of this machine can see (for example with ps).",
            fix="Pass the key through an environment variable instead (llama.cpp: LLAMA_API_KEY, "
            "vLLM: VLLM_API_KEY); with Docker, export it and pass only its name (`-e NAME`).",
        )
        for process in exposed
    ]


def check_api_key(ctx: SecurityContext) -> list[CheckResult]:
    title = "Brasskey API key"
    if ctx.api_key is None:
        return [
            CheckResult(
                id="BK-SEC-008",
                title=title,
                status=CheckStatus.SKIP,
                detail="No Brasskey API key is configured (needed only to listen "
                "beyond this machine).",
            )
        ]
    weakness = key_weakness(ctx.api_key.value)
    if weakness:
        return [
            CheckResult(
                id="BK-SEC-008",
                title=title,
                status=CheckStatus.WARN,
                detail=f"The API key from {ctx.api_key.source} is weak: {weakness}.",
                fix="Create a strong one with `brasskey run --generate-api-key`.",
            )
        ]
    return [
        CheckResult(
            id="BK-SEC-008",
            title=title,
            status=CheckStatus.PASS,
            detail=f"The API key from {ctx.api_key.source} is strong.",
        )
    ]


def check_telemetry() -> list[CheckResult]:
    return [
        CheckResult(
            id="BK-SEC-009",
            title="Telemetry",
            status=CheckStatus.PASS,
            detail="Brasskey sends no telemetry, and engines it starts run with DO_NOT_TRACK=1, "
            "HF_HUB_DISABLE_TELEMETRY=1 and VLLM_NO_USAGE_STATS=1.",
        )
    ]


def run_checks(ctx: SecurityContext) -> list[CheckResult]:
    probes = {
        candidate.listener.port: ctx.probe(probe_url(candidate.listener, candidate.label))
        for candidate in model_server_candidates(ctx)
    }
    logger.debug("Probed %d model server port(s)", len(probes))
    return [
        *check_brasskey_servers(ctx),
        *check_model_servers(ctx, probes),
        *check_browser_access(ctx, probes),
        *check_public_address(ctx),
        *check_docker(ctx),
        *check_firewall(ctx),
        *check_keys_on_command_lines(ctx),
        *check_api_key(ctx),
        *check_telemetry(),
    ]


def models_url(url: str) -> str:
    """Where to probe an endpoint the user named (base URL, /v1, or /v1/models)."""
    trimmed = url.rstrip("/")
    if trimmed.endswith("/v1/models"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/models"
    return trimmed + "/v1/models"


def check_url(url: str, probe: Callable[[str], HttpProbe | None]) -> list[CheckResult]:
    """Checks for one endpoint the user named: encryption, key, browser access."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    local = is_loopback(host)
    found = probe(models_url(url))
    results: list[CheckResult] = []
    if parsed.scheme == "https":
        if found is None:
            results.append(
                CheckResult(
                    id="BK-SEC-010",
                    title="Encryption",
                    status=CheckStatus.FAIL,
                    detail=f"Couldn't connect to {url} over HTTPS (is the "
                    "certificate valid for this name?).",
                    fix="Fix the certificate, or check the address.",
                )
            )
            return results
        results.append(
            CheckResult(
                id="BK-SEC-010",
                title="Encryption",
                status=CheckStatus.PASS,
                detail="The endpoint uses HTTPS with a valid certificate.",
            )
        )
    elif local:
        results.append(
            CheckResult(
                id="BK-SEC-010",
                title="Encryption",
                status=CheckStatus.PASS,
                detail="Plain HTTP on this machine's loopback address.",
            )
        )
    else:
        results.append(
            CheckResult(
                id="BK-SEC-010",
                title="Encryption",
                status=CheckStatus.FAIL,
                detail=f"{url} sends prompts, answers and API keys unencrypted over the network.",
                fix="Serve it over HTTPS (brasskey run --tls-cert/--tls-key) or behind a TLS "
                "reverse proxy.",
            )
        )
    if found is None:
        results.append(
            CheckResult(
                id="BK-SEC-002",
                title="Model servers need a key",
                status=CheckStatus.SKIP,
                detail=f"No answer from {models_url(url)}.",
            )
        )
        return results
    if found.open_without_key:
        status = CheckStatus.PASS if local else CheckStatus.FAIL
        detail = (
            "It answers without an API key (acceptable on loopback)."
            if local
            else "It answers anyone who can reach it, without an API key."
        )
        results.append(
            CheckResult(
                id="BK-SEC-002",
                title="Model servers need a key",
                status=status,
                detail=detail,
                fix=None if local else "Require an API key.",
            )
        )
        if found.readable_by_any_site:
            results.append(
                CheckResult(
                    id="BK-SEC-003",
                    title="Web pages can't use model servers",
                    status=CheckStatus.WARN,
                    detail="It lets any website read its answers.",
                    fix="Restrict its allowed origins.",
                )
            )
    else:
        results.append(
            CheckResult(
                id="BK-SEC-002",
                title="Model servers need a key",
                status=CheckStatus.PASS,
                detail=f"It requires an API key (HTTP {found.status} without one).",
            )
        )
    return results


def gather_context(
    os_family: str, *, keyring_get: Callable[[str, str], str | None] | None = None
) -> SecurityContext:
    """Read-only facts about this machine (and its own listening ports)."""
    return SecurityContext(
        servers=list_records(resolve_paths()),
        listeners=listening_sockets(),
        addresses=interface_addresses(),
        docker=docker_publishes(),
        firewall=firewall_state(),
        processes=own_processes(),
        api_key=resolve_api_key(None, keyring_get=keyring_get or default_keyring_get),
        probe=make_prober(),
        os_family=os_family,
    )
