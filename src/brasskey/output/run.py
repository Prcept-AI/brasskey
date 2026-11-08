# SPDX-License-Identifier: Apache-2.0
"""Views for a real ``brasskey run``: engine status, the summary before confirming, and ready."""

from pathlib import Path

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from brasskey.engines.base import EngineInstall, InstallHint
from brasskey.engines.types import EngineId
from brasskey.output.hardware import LABEL_STYLE, NOTE_STYLE, WARNING_STYLE, format_gib
from brasskey.output.plan import format_duration, format_gb
from brasskey.planner import RunPlan
from brasskey.security.policy import ServeSecurity


def install_text(engine: EngineId, install: EngineInstall | None, hint: InstallHint | None) -> str:
    if install is None:
        command = f": {hint.commands[0]}" if hint and hint.commands else ""
        return f"{engine.display_name} (not installed{command})"
    docker = " in Docker" if install.runtime == "docker" else ""
    if install.needs_setup:
        note = f": {install.setup_note}" if install.setup_note else ""
        return (
            f"{engine.display_name} {install.version}{docker} "
            f"(set up automatically on first run{note})"
        )
    if install.cuda_version is not None:
        return f"{engine.display_name} {install.version}{docker}, CUDA {install.cuda_version}"
    backends = ", ".join(backend.capitalize() for backend in install.backends) or "CPU"
    return f"{engine.display_name} {install.version}{docker}, {backends}"


def home_path(path: Path | str) -> str:
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home) :] if text.startswith(home) else text


def new_key_notice(security: ServeSecurity) -> RenderableType:
    """Shown once, right after --generate-api-key."""
    if security.api_key is None:
        return Group()
    where = (
        "It is saved in your keychain; Brasskey reads it from there next time."
        if security.stored_in_keyring
        else f"Before the next run: export {security.key_env}=<this key> (it isn't stored "
        "anywhere else)."
    )
    lines = [
        Text(
            "New API key (shown only now; give it to the clients that should have access):",
            style=LABEL_STYLE,
        ),
        Text(f"  {security.api_key.value}"),
        Text(f"  {where}", style=NOTE_STYLE),
    ]
    if security.replaced_stored_key:
        lines.append(
            Text(
                "  It replaces the key saved before: clients still using that one are refused.",
                style=WARNING_STYLE,
            )
        )
    return Group(*lines)


def run_summary(
    plan: RunPlan,
    security: ServeSecurity | None = None,
    extra_warnings: list[str] | None = None,
) -> RenderableType:
    """What is about to happen, shown before asking to confirm."""
    fit = plan.fit
    about = "about " if fit.budget.is_estimate else ""
    lines = [
        Text(
            f"{plan.model.display_name} ({plan.model.id}) with "
            f"{install_text(plan.engine, plan.install, None)}",
            style=LABEL_STYLE,
        ),
        Text(
            f"  Memory: {about}{format_gib(fit.required_per_device_bytes)} of "
            f"{format_gib(fit.budget.per_device_bytes)} ({fit.headroom_ratio:.0%} to spare), "
            "estimate"
        ),
    ]
    if fit.devices_used > 1:
        lines.append(Text(f"  GPUs: {fit.devices_used} (the model is split across them)"))
    if plan.install is not None and plan.install.needs_setup:
        lines.append(Text(f"  Setup: {plan.engine.display_name} ({plan.install.setup_note})"))
    download = plan.download
    if download is not None and download.to_download_bytes:
        timing = f", {format_duration(download.eta_seconds)}" if download.eta_seconds else ""
        lines.append(Text(f"  Download: {format_gb(download.to_download_bytes)}{timing}"))
    elif download is not None:
        lines.append(Text("  Download: nothing (already in the cache)"))
    if security is not None:
        lines.append(Text(f"  Access: {security.host}, {security.access_label}"))
    warnings = list(fit.warnings) + (list(security.warnings) if security else [])
    warnings += extra_warnings or []
    lines.extend(Text(f"  ! {warning}", style="yellow") for warning in warnings)
    return Group(*lines)


def ready_view(
    plan: RunPlan,
    *,
    base_url: str,
    served_model_name: str,
    startup_seconds: float,
    log_path: Path,
    notes: list[str],
    security: ServeSecurity | None = None,
) -> RenderableType:
    auth = ""
    if security is not None and security.api_key is not None:
        source = security.api_key.source
        variable = source[4:] if source.startswith("env:") else None
        token = f"${variable}" if variable else "<your key>"
        auth = f"-H 'Authorization: Bearer {token}' "
    curl = (
        f"curl {base_url}/chat/completions {auth}-H 'Content-Type: application/json' "
        f'-d \'{{"model": "{served_model_name}", "messages": '
        '[{"role": "user", "content": "Hello!"}]}\''
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style=LABEL_STYLE, no_wrap=True)
    table.add_column()
    table.add_row(Text("API"), Text(f"{base_url}  (OpenAI-compatible)"))
    table.add_row(Text("Model name"), Text(served_model_name))
    if security is not None:
        access = f"{security.access_label}" + (", HTTPS" if security.tls_cert else "")
        table.add_row(Text("Access"), Text(access))
        if security.wildcard:
            table.add_row(Text("Listening"), Text(f"all network interfaces ({security.host})"))
    table.add_row(Text("Engine"), Text(install_text(plan.engine, plan.install, None)))
    table.add_row(Text("Started in"), Text(f"{startup_seconds:.1f}s (measured)"))
    table.add_row(Text("Log"), Text(home_path(log_path), style=NOTE_STYLE))
    parts: list[RenderableType] = [
        Text(f"{plan.model.display_name} is running.", style=LABEL_STYLE),
        table,
        Text(""),
        Text("Try it from another terminal:", style=LABEL_STYLE),
        Text(f"  {curl}"),
    ]
    parts += [Text(f"i {note}", style=NOTE_STYLE) for note in notes]
    if security is not None:
        parts += [Text(f"! {warning}", style="yellow") for warning in security.warnings]
    parts += [Text(""), Text("Press Ctrl-C to stop.", style=NOTE_STYLE)]
    return Group(*parts)
