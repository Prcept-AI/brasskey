# SPDX-License-Identifier: Apache-2.0
"""``brasskey doctor``: check this machine.

Plain ``doctor`` shows the hardware (read-only probe), the engines Brasskey can start and whether
they are installed (with the install command if not), and servers Brasskey is running.
``doctor --security [URL]`` checks exposure: Brasskey's servers, other local model servers
(Ollama, LM Studio, ...), public addresses, Docker publishing, the firewall, API keys on command
lines (ADR 0009). It exits with code 3 when any check fails.
"""

import logging
from typing import Annotated

import typer
from pydantic import JsonValue
from rich.console import Group
from rich.text import Text

from brasskey.cli.options import DebugFlag, JsonFlag, YesFlag, apply_flags
from brasskey.engines.capabilities import check_cuda_driver, check_engine, engine_preference
from brasskey.engines.registry import adapter_for, detect_engines, refine_hardware
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError, ExitCode
from brasskey.hardware import probe_hardware
from brasskey.output.hardware import LABEL_STYLE, hardware_view
from brasskey.output.next_step import NextStep
from brasskey.output.result import CommandResult
from brasskey.output.run import install_text
from brasskey.output.security import security_view, summary_counts
from brasskey.paths import resolve_paths
from brasskey.runtime.state import list_records
from brasskey.security.doctor import (
    CheckStatus,
    SecurityContext,
    check_url,
    gather_context,
    run_checks,
)
from brasskey.security.netinfo import Prober, make_prober

logger = logging.getLogger(__name__)


def gather_security_context(os_family: str) -> SecurityContext:
    """Seam for tests: the read-only facts behind --security."""
    return gather_context(os_family)


def make_url_prober() -> Prober:
    return make_prober()


def doctor_command(
    url: Annotated[
        str | None,
        typer.Argument(
            help="With --security: an endpoint to check instead of this machine.",
            show_default=False,
        ),
    ] = None,
    security: Annotated[
        bool,
        typer.Option("--security", help="Check bind address, auth, TLS and network exposure."),
    ] = False,
    json_output: JsonFlag = False,
    debug: DebugFlag = False,
    yes: YesFlag = False,
) -> CommandResult:
    """Check this machine: hardware, engines and running servers, or security with --security."""
    apply_flags("doctor", json_output=json_output, debug=debug, yes=yes)
    if url is not None and not security:
        raise BrasskeyError(
            "Checking an endpoint is part of the security checks.",
            fix=f"Run `brasskey doctor --security {url}`.",
            code="BK-E-USAGE",
            exit_code=ExitCode.USAGE,
        )
    if security:
        return _security(url)

    installs = detect_engines()
    report = refine_hardware(probe_hardware(), installs)
    servers = list_records(resolve_paths())
    logger.debug(
        "doctor: %d accelerator(s), %d issue(s), %d engine(s) installed, %d server(s) running",
        len(report.accelerators),
        len(report.issues),
        sum(1 for install in installs.values() if install is not None),
        len(servers),
    )

    engine_lines: list[Text] = [Text("Engines", style=LABEL_STYLE)]
    hints: dict[str, JsonValue] = {}
    install_commands: dict[EngineId, str] = {}
    for engine, install in installs.items():
        support = check_engine(engine, report)
        if install is None and not support.supported:
            engine_lines.append(
                Text(f"  {engine.display_name}: not for this machine ({support.reasons[0]})")
            )
            continue
        adapter = adapter_for(engine)
        hint = adapter.install_hint(report) if adapter is not None and install is None else None
        if hint is not None:
            hints[engine.value] = hint.model_dump(mode="json")
            if hint.commands:
                install_commands[engine] = hint.commands[0]
        engine_lines.append(Text(f"  {install_text(engine, install, hint)}"))
        if install is not None:
            for warning in check_cuda_driver(engine, install.cuda_version, report).notes:
                engine_lines.append(Text(f"    ! {warning}", style="yellow"))
    # Suggest installing the engine Brasskey would pick here first (vLLM on NVIDIA Linux, ...).
    preferred = [engine for engine in engine_preference(report) if engine in install_commands]
    missing_command = (
        install_commands[preferred[0]] if preferred else next(iter(install_commands.values()), None)
    )

    server_lines: list[Text] = []
    if servers:
        server_lines = [Text(""), Text("Running servers", style=LABEL_STYLE)]
        server_lines += [
            Text(
                f"  {server.model_id} at {server.base_url} ({server.engine}, pid {server.pid}, "
                f"since {server.started_at:%Y-%m-%d %H:%M} UTC)"
            )
            for server in servers
        ]

    data: dict[str, JsonValue] = {
        "hardware": report.model_dump(mode="json"),
        "fingerprint": report.fingerprint(),
        "engines": {
            engine.value: install.model_dump(mode="json") if install else None
            for engine, install in installs.items()
        },
        "install_hints": hints,
        "servers": [server.model_dump(mode="json") for server in servers],
    }
    if missing_command is not None:
        next_step = NextStep(command=missing_command, hint="install an engine")
    else:
        next_step = NextStep(command="brasskey run --dry-run", hint="see what Brasskey would run")
    view = Group(hardware_view(report), Text(""), *engine_lines, *server_lines)
    return CommandResult(command="doctor", data=data, next_step=next_step, human=view)


def _security(url: str | None) -> CommandResult:
    if url is not None:
        results = check_url(url, make_url_prober())
    else:
        family = probe_hardware().os.family
        results = run_checks(gather_security_context(family))
    counts = summary_counts(results)
    logger.debug("Security checks: %s", dict(counts))
    failed = counts[CheckStatus.FAIL] > 0
    data: dict[str, JsonValue] = {
        "target": url or "this machine",
        "checks": [result.model_dump(mode="json") for result in results],
        "summary": {status.value: counts[status] for status in CheckStatus},
        "passed": not failed,
    }
    again = f"brasskey doctor --security {url}" if url else "brasskey doctor --security"
    next_step = (
        NextStep(command=again, hint="check again after fixing the failures")
        if failed
        else NextStep(command="brasskey run", hint="start a model")
    )
    return CommandResult(
        command="doctor",
        data=data,
        next_step=next_step,
        human=security_view(results, target=url),
        exit_code=ExitCode.CHECK_FAILED if failed else ExitCode.OK,
    )
