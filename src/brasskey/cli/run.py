# SPDX-License-Identifier: Apache-2.0
"""``brasskey run``: start a local OpenAI-compatible server for a model that fits this machine.

Flow: probe hardware -> detect engines -> plan (from a model name, ``brasskey.yaml`` or
``brasskey.lock``) -> check address, API key and TLS -> estimate the download -> confirm ->
download and verify -> start the engine on a private loopback port -> wait until healthy -> start
the Brasskey proxy on the requested address -> serve in the foreground until Ctrl-C.
``--dry-run`` stops after planning.

Where the plan comes from (ADR 0002, ADR 0006):
- a model named on the command line (files in this folder are ignored, and said so);
- otherwise ``brasskey.lock`` in this folder is replayed (unless ``brasskey.yaml`` changed since);
- otherwise ``brasskey.yaml`` is planned; otherwise Brasskey picks a quick-start model.
Files are written only with ``--save``, or for the lock when ``brasskey.yaml`` is present.

Exposure (ADR 0009): clients always reach the engine through the Brasskey proxy. Listening on
anything but loopback requires an API key (``BRASSKEY_API_KEY``, the keyring, or
``--generate-api-key``); TLS is optional and recommended off-loopback.
"""

import logging
import os
import re
import shlex
import socket
import time
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated

import httpx2
import typer
from pydantic import JsonValue

from brasskey import __version__
from brasskey.cli.options import DebugFlag, JsonFlag, YesFlag, apply_flags
from brasskey.config.intent import INTENT_FILE, GoalIntent, Intent, ServeIntent, TlsIntent
from brasskey.config.io import intent_sha256, read_intent, read_lock, write_intent, write_lock
from brasskey.config.lock import LOCK_FILE
from brasskey.engines.base import (
    INTERNAL_KEY_ENV,
    EngineAdapter,
    EngineInstall,
    EngineNotInstalledError,
    EngineStartError,
    ServeOptions,
)
from brasskey.engines.capabilities import GPU_SERVER_ENGINES, check_cuda_driver, choose_gpus
from brasskey.engines.registry import (
    adapter_for,
    detect_engines,
    refine_hardware,
    reported_gpu_backends,
    runnable_engines,
)
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError, ExitCode
from brasskey.hardware import HardwareReport, probe_hardware
from brasskey.hardware.types import AcceleratorKind
from brasskey.log import register_secret
from brasskey.models.catalog import load_catalog
from brasskey.models.download import estimate_download, measure_bandwidth
from brasskey.models.fetch import DownloadError, fetch_artifact
from brasskey.models.fit import MIN_CONTEXT
from brasskey.models.hub import HubClient, RealHub
from brasskey.output.confirm import confirm
from brasskey.output.next_step import NextStep
from brasskey.output.plan import format_gb, plan_view
from brasskey.output.progress import fetch_display
from brasskey.output.render import emit_error, emit_result, stderr_console
from brasskey.output.result import CommandResult
from brasskey.output.run import install_text, new_key_notice, ready_view, run_summary
from brasskey.output.settings import OutputSettings
from brasskey.paths import resolve_paths
from brasskey.planner import RunPlan, RunRequest, plan_run
from brasskey.planner.replay import lock_from_plan, plan_from_lock
from brasskey.runtime.foreground import serve_until_stopped
from brasskey.runtime.ports import choose_port
from brasskey.runtime.proxy import ProxyConfig, ProxyServer, ProxyStartError, create_app
from brasskey.runtime.setup import run_engine_setup
from brasskey.runtime.state import ServerRecord, process_start_time, remove_record, write_record
from brasskey.runtime.supervisor import ProcessSupervisor, ServerHandle
from brasskey.security.apikeys import (
    DEFAULT_KEY_ENV,
    ApiKeyError,
    default_keyring_get,
    default_keyring_set,
    generate_api_key,
)
from brasskey.security.policy import KEY_REQUIRED_CODE, ServeSecurity, decide_security

logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CONTEXT = 8192
DEFAULT_HOST = "127.0.0.1"
HEALTH_TIMEOUT_S = 2.0
WATCH_INTERVAL_S = 0.25


# --- seams for tests ------------------------------------------------------------------------------


def make_hub(*, offline: bool) -> HubClient:
    return RealHub(offline=offline)


def make_supervisor(
    extra_env: Mapping[str, str] | None = None, drop_env: Collection[str] = ()
) -> ProcessSupervisor:
    """The supervisor, with the engine's environment: no user API key, plus the engine's own."""
    env = {name: value for name, value in os.environ.items() if name not in drop_env}
    env.update(extra_env or {})
    return ProcessSupervisor(resolve_paths(), env=env)


def make_proxy(
    config: ProxyConfig, *, host: str, port: int, tls_cert: Path | None, tls_key: Path | None
) -> ProxyServer:
    return ProxyServer(create_app(config), host=host, port=port, tls_cert=tls_cert, tls_key=tls_key)


def make_http_client() -> httpx2.Client:
    # trust_env=False: never send local health checks through a proxy from the environment.
    return httpx2.Client(timeout=HEALTH_TIMEOUT_S, trust_env=False)


# The OS keyring (tests replace these so they never touch a real keychain).
keyring_reader = default_keyring_get
keyring_writer = default_keyring_set


def free_loopback_port() -> int:
    """A free port on 127.0.0.1 for the engine (only the proxy talks to it)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- resolving what to run ------------------------------------------------------------------------


@dataclass
class Resolved:
    plan: RunPlan
    intent: Intent | None  # from brasskey.yaml, or built from the command line for --save
    intent_from_file: bool
    lock_replayed: bool
    host: str
    port: int | None
    api_key_env: str | None
    tls_cert: Path | None
    tls_key: Path | None
    write_lock_file: bool
    notes: list[str] = field(default_factory=list[str])

    @property
    def source(self) -> str:
        if self.lock_replayed:
            return "brasskey.lock"
        return "brasskey.yaml" if self.intent_from_file else "command line"


@dataclass(frozen=True)
class CliServe:
    """Serving options given on the command line (None: not given)."""

    host: str | None = None
    port: int | None = None
    api_key_env: str | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None


def _run_again(
    model: str | None,
    *,
    engine: EngineId | None,
    context: int | None,
    cli: CliServe,
    extra: Sequence[str] = (),
) -> str:
    """The ``brasskey run`` command that repeats this one (for ``Next:`` lines).

    It keeps the serving options the user typed (address, port, key variable, TLS files), so the
    next step reproduces what was just planned instead of silently falling back to defaults.
    ``model`` is None when brasskey.yaml / brasskey.lock in this folder supplied the plan.
    """
    parts = ["brasskey", "run"]
    if model is not None:
        parts.append(model)
    if engine is not None:
        parts += ["--engine", engine.value]
    if context is not None:
        parts += ["--context", str(context)]
    if cli.host is not None:
        parts += ["--host", cli.host]
    if cli.port is not None:
        parts += ["--port", str(cli.port)]
    if cli.api_key_env is not None:
        parts += ["--api-key-env", cli.api_key_env]
    if cli.tls_cert is not None and cli.tls_key is not None:
        parts += ["--tls-cert", str(cli.tls_cert), "--tls-key", str(cli.tls_key)]
    return shlex.join([*parts, *extra])


def _resolve(
    folder: Path,
    *,
    model: str | None,
    engine: EngineId | None,
    context: int | None,
    cli: CliServe,
    save: bool,
    hw: HardwareReport,
    hub: HubClient,
    installs: Mapping[EngineId, EngineInstall | None],
) -> Resolved:
    runnable = runnable_engines()
    installed = [engine_id for engine_id, found in installs.items() if found is not None]
    gpu_backends = reported_gpu_backends(installs)  # e.g. a CPU-only llama.cpp: plan for the CPU
    notes: list[str] = []
    file_intent = None
    lock = None
    if model is None:
        file_intent = read_intent(folder)
        lock = read_lock(folder)
    elif (folder / INTENT_FILE).exists() or (folder / LOCK_FILE).exists():
        notes.append(
            "You named a model, so brasskey.yaml / brasskey.lock in this folder are ignored."
        )
    overridden = engine is not None or context is not None
    serve = (file_intent or Intent()).serve
    tls = serve.tls
    tls_cert = cli.tls_cert or (tls.cert_file if tls else None)
    tls_key = cli.tls_key or (tls.key_file if tls else None)

    if lock is not None and not overridden:
        if file_intent is None or lock.intent_sha256 == intent_sha256(file_intent):
            plan = plan_from_lock(lock, hw, runnable=runnable, gpu_backends=gpu_backends)
            return Resolved(
                plan=plan,
                intent=file_intent,
                intent_from_file=file_intent is not None,
                lock_replayed=True,
                host=cli.host or lock.serve.host,
                port=cli.port if cli.port is not None else lock.serve.port,
                api_key_env=cli.api_key_env or lock.serve.api_key_env,
                tls_cert=tls_cert,
                tls_key=tls_key,
                write_lock_file=False,
                notes=notes,
            )
        notes.append(
            "brasskey.yaml changed since brasskey.lock was written, so it was planned again."
        )

    base = file_intent or Intent()
    request = RunRequest(
        model=model or base.model,
        engine=engine or base.engine,
        context_length=context or base.goal.context_length,
        concurrency=base.goal.concurrency,
    )
    plan = plan_run(
        request,
        hw,
        load_catalog(),
        hub=hub,
        cached_bytes=hub.cached_bytes,
        runnable=runnable,
        installed=installed,
        gpu_backends=gpu_backends,
    )
    host = cli.host or serve.host
    intent = file_intent or Intent(
        model=plan.model.id,
        engine=request.engine,
        goal=GoalIntent(context_length=request.context_length, concurrency=request.concurrency),
        serve=ServeIntent(
            host=host,
            port=cli.port,
            api_key_env=cli.api_key_env,
            tls=TlsIntent(cert_file=tls_cert, key_file=tls_key) if tls_cert and tls_key else None,
        ),
    )
    return Resolved(
        plan=plan,
        intent=intent,
        intent_from_file=file_intent is not None,
        lock_replayed=False,
        host=host,
        port=cli.port if cli.port is not None else serve.port,
        api_key_env=cli.api_key_env or serve.api_key_env,
        tls_cert=tls_cert,
        tls_key=tls_key,
        write_lock_file=save or (file_intent is not None and not overridden and model is None),
        notes=notes,
    )


def _save_files(folder: Path, resolved: Resolved, save: bool) -> list[str]:
    """Write brasskey.yaml for --save (never overwriting an existing one). Returns notes."""
    if save and not resolved.intent_from_file and resolved.intent is not None:
        write_intent(folder, resolved.intent)
        return [f"Wrote {INTENT_FILE}."]
    return []


def _write_lock(
    folder: Path,
    plan: RunPlan,
    hw: HardwareReport,
    resolved: Resolved,
    security: ServeSecurity,
    **measured: object,
) -> Path:
    lock = lock_from_plan(
        plan,
        hw,
        intent=resolved.intent,
        host=security.host,
        port=resolved.port,
        api_key_env=resolved.api_key_env,
        api_key_source=security.api_key.source if security.api_key else None,
        tls=security.tls_cert is not None,
        **measured,  # type: ignore[arg-type]
    )
    return write_lock(folder, lock)


# --- the command ----------------------------------------------------------------------------------


@contextmanager
def _status(settings: OutputSettings, message: str) -> Generator[None, None, None]:
    if settings.json:
        yield
        return
    with stderr_console().status(message):
        yield


def _server_id(model_id: str, port: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", model_id).strip("-").lower()
    return f"{slug}-{port}-{datetime.now(UTC):%Y%m%dT%H%M%S}"


def run_command(
    model: Annotated[
        str | None,
        typer.Argument(
            help="Catalog id such as qwen3-4b, or hf:org/name[:QUANT]. Omit to let Brasskey pick.",
            show_default=False,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the plan without downloading or starting anything."),
    ] = False,
    engine: Annotated[
        EngineId | None,
        typer.Option("--engine", case_sensitive=False, help="Engine to use (default: best here)."),
    ] = None,
    context: Annotated[
        int | None,
        typer.Option(
            "--context", min=MIN_CONTEXT, help="Context length in tokens (default: 8192)."
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            "--host", help="Address to listen on (default 127.0.0.1; others need an API key)."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Port (default: 8000 or the next free)."),
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option(
            "--api-key-env", help=f"Variable holding the API key (default: {DEFAULT_KEY_ENV})."
        ),
    ] = None,
    generate_api_key_flag: Annotated[
        bool,
        typer.Option(
            "--generate-api-key", help="Create a strong API key, keep it in the keychain, show it."
        ),
    ] = False,
    tls_cert: Annotated[
        Path | None, typer.Option("--tls-cert", help="PEM certificate to serve HTTPS.")
    ] = None,
    tls_key: Annotated[
        Path | None, typer.Option("--tls-key", help="PEM private key to serve HTTPS.")
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save", help="Write brasskey.yaml and brasskey.lock in this folder."),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Don't contact Hugging Face (cached models only)."),
    ] = False,
    json_output: JsonFlag = False,
    debug: DebugFlag = False,
    yes: YesFlag = False,
) -> CommandResult | int:
    """Start a local OpenAI-compatible server for a model that fits this machine."""
    settings = apply_flags("run", json_output=json_output, debug=debug, yes=yes)
    logger.debug("run requested (model=%s, dry_run=%s)", model or "<auto>", dry_run)
    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").lower() in TRUTHY
    folder = Path.cwd()

    installs = detect_engines()
    hardware = refine_hardware(probe_hardware(), installs)
    hub = make_hub(offline=offline)
    cli = CliServe(
        host=host, port=port, api_key_env=api_key_env, tls_cert=tls_cert, tls_key=tls_key
    )
    resolved = _resolve(
        folder,
        model=model,
        engine=engine,
        context=context,
        cli=cli,
        save=save,
        hw=hardware,
        hub=hub,
        installs=installs,
    )
    # With brasskey.yaml / brasskey.lock in charge, plain `brasskey run` repeats this plan.
    from_files = resolved.intent_from_file or resolved.lock_replayed
    shown_model = None if from_files else resolved.plan.model.id
    again = partial(_run_again, shown_model, engine=engine, context=context, cli=cli)
    try:
        security = decide_security(
            resolved.host or DEFAULT_HOST,
            api_key_env=resolved.api_key_env,
            generate=generate_api_key_flag,
            tls_cert=resolved.tls_cert,
            tls_key=resolved.tls_key,
            keyring_get=keyring_reader,
            keyring_set=keyring_writer,
            preview=dry_run,  # a dry run never creates or saves a key
        )
    except ApiKeyError as error:
        if error.code == KEY_REQUIRED_CODE:
            dry = ("--dry-run",) if dry_run else ()
            error.next_step = NextStep(
                command=again(extra=(*dry, "--generate-api-key")),
                hint="plan it with a new key" if dry_run else "create a key",
            )
        raise
    if security.generated and security.api_key is not None and not settings.json:
        stderr_console().print(new_key_notice(security))
    install = installs.get(resolved.plan.engine)
    download = estimate_download(
        resolved.plan.fit.artifact, hub=hub, bandwidth=None if offline else measure_bandwidth
    )
    plan = resolved.plan.with_download(download).with_install(install)
    adapter = adapter_for(plan.engine)
    if adapter is None:  # the planner only picks runnable engines; this guards future edits
        raise BrasskeyError(f"No adapter for {plan.engine.display_name} in this build.")

    if dry_run:
        # The key is created when it really starts, so starting keeps --generate-api-key.
        start = again(extra=("--generate-api-key",) if security.key_pending else ())
        return _dry_run(plan, resolved, security, hardware, adapter, folder, save, start)
    return _start_and_serve(
        plan, resolved, security, hardware, adapter, install, hub, settings, folder, save
    )


def _engine_warnings(plan: RunPlan, hw: HardwareReport) -> list[str]:
    """Warnings about the installed engine on this hardware (e.g. an old NVIDIA driver)."""
    cuda = plan.install.cuda_version if plan.install is not None else None
    return list(check_cuda_driver(plan.engine, cuda, hw).notes)


def _gpu_indices(plan: RunPlan, hw: HardwareReport) -> tuple[int, ...]:
    """The GPUs vLLM / SGLang get: as many as the plan splits the model across, most free first."""
    if plan.engine not in GPU_SERVER_ENGINES:
        return ()
    return choose_gpus(hw.accelerators_of(AcceleratorKind.NVIDIA_GPU), plan.fit.devices_used)


def _key_data(security: ServeSecurity) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "listen_host": security.host,
        "auth": "api_key" if security.api_key else "none",
        "api_key_source": security.api_key.source if security.api_key else None,
        "tls": security.tls_cert is not None,
        "warnings": list(security.warnings),
        "new_key_on_start": security.key_pending,  # dry run: created when it really starts
        "replaced_stored_key": security.replaced_stored_key,
    }
    if security.generated and security.api_key is not None:
        data["generated_api_key"] = security.api_key.value  # the one time it is shown
    return data


def _dry_run(
    plan: RunPlan,
    resolved: Resolved,
    security: ServeSecurity,
    hw: HardwareReport,
    adapter: EngineAdapter,
    folder: Path,
    save: bool,
    start_command: str,
) -> CommandResult:
    notes = list(resolved.notes) + _save_files(folder, resolved, save)
    if resolved.write_lock_file:
        _write_lock(folder, plan, hw, resolved, security)
        notes.append(f"Wrote {LOCK_FILE} (planned; not started yet).")
    hint = adapter.install_hint(hw) if plan.install is None else None
    if hint is not None and hint.commands:
        next_step = NextStep(command=hint.commands[0], hint=f"install {plan.engine.display_name}")
    else:
        next_step = NextStep(command=start_command, hint="start it")
    data: dict[str, JsonValue] = {
        "dry_run": True,
        "plan_source": resolved.source,
        "plan": plan.model_dump(mode="json", exclude={"model": {"artifacts"}}),
        "engine_installed": plan.install is not None,
        "install_hint": hint.model_dump(mode="json") if hint else None,
        "security": _key_data(security),
        "engine_warnings": list(_engine_warnings(plan, hw)),
        "gpu_indices": list(_gpu_indices(plan, hw)),
        "notes": list(notes),
        "hardware_fingerprint": hw.fingerprint(),
    }
    view = plan_view(
        plan,
        engine_line=install_text(plan.engine, plan.install, hint),
        access_line=f"{security.host}, {security.access_label}"
        + (", HTTPS" if security.tls_cert else ""),
        extra_warnings=list(security.warnings) + _engine_warnings(plan, hw),
    )
    return CommandResult(command="run", data=data, next_step=next_step, human=view)


def _watch(handle: ServerHandle, proxy: ProxyServer) -> Callable[[], int]:
    """Wait until the engine exits (its code) or the proxy dies (an error)."""

    def wait() -> int:
        while True:
            code = handle.process.poll()
            if code is not None:
                return code
            if not proxy.alive:
                raise ProxyStartError(
                    "The Brasskey proxy stopped unexpectedly.",
                    fix="Run again with --debug and report it if it repeats.",
                )
            time.sleep(WATCH_INTERVAL_S)

    return wait


def _start_and_serve(
    plan: RunPlan,
    resolved: Resolved,
    security: ServeSecurity,
    hw: HardwareReport,
    adapter: EngineAdapter,
    install: EngineInstall | None,
    hub: HubClient,
    settings: OutputSettings,
    folder: Path,
    save: bool,
) -> int:
    name = plan.engine.display_name
    if install is None:
        hint = adapter.install_hint(hw)
        command = hint.commands[0] if hint.commands else None
        raise EngineNotInstalledError(
            f"{name} isn't installed, so Brasskey can't start {plan.model.display_name}.",
            fix=f"{hint.summary}" + (f" Run: {command}" if command else ""),
            next_step=NextStep(command=command, hint=f"install {name}") if command else None,
        )
    download = plan.download
    if download is not None and download.fits_on_disk is False:
        raise DownloadError(
            f"Not enough disk space: {format_gb(download.to_download_bytes)} to download but only "
            f"{format_gb(download.disk_free_bytes or 0)} free in {download.cache_dir}.",
            fix="Free some space, or set HF_HOME to a bigger disk.",
        )
    # Before the prompt and any download: a busy port must not cost a long download first.
    public_port, port_note = choose_port(security.host, resolved.port)

    if not settings.json:
        stderr_console().print(run_summary(plan, security, _engine_warnings(plan, hw)))
    remaining = download.to_download_bytes if download is not None else 0
    steps: list[str] = []
    if install.needs_setup:
        steps.append(f"set up {name} ({install.setup_note or 'one time'})")
    if remaining:
        steps.append(f"download {format_gb(remaining)}")
    steps.append(f"start {plan.model.display_name} with {name}")
    question = ", ".join(steps[:-1]) + " and " + steps[-1] if len(steps) > 1 else steps[0]
    question = question[0].upper() + question[1:] + "?"
    if not confirm(question, settings=settings):
        raise BrasskeyError(
            "Cancelled: nothing was downloaded or started.",
            code="BK-E-CANCELLED",
            next_step=NextStep(command=f"brasskey run {plan.model.id} --dry-run", hint="review it"),
        )

    if install.needs_setup:
        with _status(settings, f"Setting up {name} ({install.setup_note or 'one time'})..."):
            run_engine_setup(install)
        install = adapter.detect() or install
        plan = plan.with_install(install)

    with fetch_display(enabled=not settings.json) as display:
        local = fetch_artifact(
            plan.fit.artifact,
            hub=hub,
            on_progress=display.on_download,
            on_verify=display.on_verify,
        )

    engine_port = free_loopback_port()
    engine_key = generate_api_key() if adapter.supports_api_key else None
    if engine_key is not None:
        register_secret(engine_key)
    options = ServeOptions(
        host="127.0.0.1",  # the engine is private; only the proxy listens on security.host
        port=engine_port,
        served_model_name=plan.model.id,
        context_length=plan.fit.context_length,
        parallel=plan.fit.concurrency,
        api_key_env=INTERNAL_KEY_ENV if engine_key else None,
        cache_budget_bytes=plan.fit.kv_cache_bytes,
        tensor_parallel=plan.fit.devices_used if plan.engine in GPU_SERVER_ENGINES else 1,
        gpu_indices=_gpu_indices(plan, hw),
    )
    spec = adapter.build_launch(install, local, options, hw)
    user_key_vars = {resolved.api_key_env or DEFAULT_KEY_ENV}
    supervisor = make_supervisor(
        extra_env={INTERNAL_KEY_ENV: engine_key} if engine_key else None, drop_env=user_key_vars
    )
    server_id = _server_id(plan.model.id, public_port)
    handle = supervisor.start(spec, server_id=server_id, engine_name=name)
    proxy: ProxyServer | None = None
    record_written = False
    exit_code: int | None = None
    try:
        weights = format_gb(plan.fit.weights_bytes)
        with (
            make_http_client() as client,
            _status(settings, f"Starting {name} (loading {weights} into memory)..."),
        ):
            seconds = supervisor.wait_until_ready(handle, adapter, client=client)

        proxy = make_proxy(
            ProxyConfig(
                upstream=f"http://127.0.0.1:{engine_port}",
                served_model=options.served_model_name,
                api_key=security.api_key.value if security.api_key else None,
                upstream_api_key=engine_key,
                allowed_origins=security.origins(public_port),
            ),
            host=security.host,
            port=public_port,
            tls_cert=security.tls_cert,
            tls_key=security.tls_key,
        )
        proxy.start()
        base_url = f"{security.scheme}://{security.client_host}:{public_port}/v1"

        write_record(
            supervisor.paths,
            ServerRecord(
                id=server_id,
                pid=handle.pid,
                process_started_at=process_start_time(handle.pid) or 0.0,
                engine=plan.engine.value,
                engine_version=install.version,
                model_id=plan.model.id,
                host=security.host,
                port=public_port,
                engine_port=engine_port,
                tls=security.tls_cert is not None,
                base_url=base_url,
                auth="api_key" if security.api_key else "none",
                started_at=datetime.now(UTC),
                log_path=str(handle.log_path),
                brasskey_version=__version__,
            ),
        )
        record_written = True

        notes = list(resolved.notes) + ([port_note] if port_note else [])
        notes += _save_files(folder, resolved, save)
        lock_written = None
        if resolved.write_lock_file:
            lock_written = str(
                _write_lock(
                    folder, plan, hw, resolved, security, launch_args=spec.argv,
                    model_file=str(local.primary_file), startup_seconds=seconds,
                    placeholders=spec.placeholders,
                )
            )  # fmt: skip
            notes.append(f"Wrote {LOCK_FILE} (with the measured start time).")

        data: dict[str, JsonValue] = {
            "status": "ready",
            "plan_source": resolved.source,
            "base_url": base_url,
            "model": options.served_model_name,
            "engine": plan.engine.value,
            "engine_version": install.version,
            "pid": handle.pid,
            "port": public_port,
            "engine_port": engine_port,
            "engine_runtime": install.runtime,
            "gpu_indices": list(options.gpu_indices),
            "startup_seconds": round(seconds, 2),
            "log_path": str(handle.log_path),
            "lock_path": lock_written,
            "notes": list(notes),
            **_key_data(security),
        }
        emit_result(
            CommandResult(
                command="run",
                data=data,
                next_step=NextStep(
                    command=f"curl {base_url}/models"
                    + (" -H 'Authorization: Bearer <your key>'" if security.api_key else ""),
                    hint="check it from another terminal",
                ),
                human=ready_view(
                    plan,
                    base_url=base_url,
                    served_model_name=options.served_model_name,
                    startup_seconds=seconds,
                    log_path=handle.log_path,
                    notes=notes,
                    security=security,
                ),
            ),
            settings,
        )
        exit_code = serve_until_stopped(handle, supervisor, wait=_watch(handle, proxy))
    finally:
        if proxy is not None:
            proxy.stop()
        if handle.process.poll() is None:
            supervisor.stop(handle)
        if record_written:
            remove_record(supervisor.paths, server_id)

    if exit_code is None:
        if not settings.json:
            stderr_console().print(f"Stopped {plan.model.display_name} ({name}).")
        return int(ExitCode.OK)
    error = supervisor.failure(
        handle, adapter, exit_code, f"stopped unexpectedly (exit code {exit_code})"
    )
    if settings.json:  # the ready document was already printed; keep stdout to one document
        plain = OutputSettings(json=False, debug=settings.debug, yes=settings.yes, command="run")
        emit_error(
            command="run",
            code=error.code,
            message=error.message,
            fix=error.fix,
            next_step=None,
            settings=plain,
        )
        return int(ExitCode.ERROR)
    raise EngineStartError(error.message, fix=error.fix)
