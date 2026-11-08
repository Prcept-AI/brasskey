# SPDX-License-Identifier: Apache-2.0
"""MLX adapter: runs ``mlx_lm.server`` (mlx-lm, MIT) as a separate process on Apple Silicon.

Where it comes from (ADR 0002, ADR 0007):
- ``mlx_lm.server`` already on PATH (the user installed mlx-lm): used as is;
- otherwise a pinned, isolated environment run by ``uv tool run``: mlx-lm 0.32.2, its
  dependencies resolved as of a fixed date, on uv's own Python 3.12. The first use downloads it
  (about 70 MB) after the user confirms; later runs start offline from uv's cache.

Model naming: mlx_lm.server loads whatever model name a request asks for, mapping only the name it
was started with to the loaded model. Brasskey starts it with ``--model <served name>``, a link in
a private working folder, so the advertised name (or none) uses the loaded model.

Defaults chosen for a private server that stays within the memory plan:
- ``--allowed-origins`` set to the server's own address (mlx-lm's default ``*`` would let any web
  page in a browser read responses);
- ``--decode-concurrency`` / ``--prompt-concurrency`` set to the planned concurrency (defaults 32
  and 8 could allocate far more KV cache than planned);
- ``--prompt-cache-bytes`` capped at the planned KV-cache size (default: unlimited);
- ``--log-level INFO``: request bodies and generated text are only logged at DEBUG.

mlx-lm's server has no API keys (it says itself it "only implements basic security checks"), so
it listens on a private loopback port and the Brasskey proxy enforces keys (ADR 0009).
"""

import json
import logging
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from brasskey.engines.base import (
    ENGINE_BASE_ENV,
    EngineAdapter,
    EngineDevice,
    EngineInstall,
    InstallHint,
    LaunchSpec,
    LocalModel,
    ServeOptions,
)
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandRunner, run_command
from brasskey.hardware.types import HardwareReport

logger = logging.getLogger(__name__)

SERVER = "mlx_lm.server"
MLX_LM_VERSION = "0.32.2"
RESOLVE_CUTOFF = "2026-11-01T00:00:00Z"  # dependencies resolved as of this date: reproducible
UV_PYTHON = "3.12"
SETUP_NOTE = "about 70 MB, one time"
DOCS_URL = "https://github.com/ml-explore/mlx-lm"
FALLBACK_DIRS = (Path.home() / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))

# Flags of mlx_lm.server 0.32.2 (from its --help); used for the pinned uv environment.
PINNED_FLAGS = frozenset(
    {
        "--model", "--host", "--port", "--allowed-origins", "--log-level", "--decode-concurrency",
        "--prompt-concurrency", "--prompt-cache-size", "--prompt-cache-bytes", "--max-tokens",
        "--chat-template-args", "--adapter-path", "--draft-model", "--num-draft-tokens",
        "--trust-remote-code", "--chat-template", "--use-default-chat-template", "--temp",
        "--top-p", "--top-k", "--min-p", "--prefill-step-size", "--kv-bits", "--kv-group-size",
        "--quantized-kv-start", "--pipeline", "--help",
    }
)  # fmt: skip

# Runs inside the engine's environment (never in Brasskey's): versions and Metal limits.
PROBE = (
    "import json, importlib.metadata as m, mlx.core as mx; "
    "info = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info(); "
    "print(json.dumps({'mlx_lm': m.version('mlx-lm'), 'mlx': m.version('mlx'), 'device': info}))"
)

_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")

# (log fragments, all of them must appear?, message, fix), checked in order.
_FAILURES: tuple[tuple[tuple[str, ...], bool, str, str], ...] = (
    (
        ("insufficient memory", "outofmemory", "out of memory", "[metal::malloc]",
         "resource limit"),
        False,
        "MLX ran out of memory while loading or running the model.",
        "Use a smaller model or a shorter --context (`brasskey run --dry-run` shows what fits), "
        "and close other apps.",
    ),
    (
        ("address already in use",),
        False,
        "The server port is already in use.",
        "Pass --port with a free port, or stop the other server.",
    ),
    (
        ("model type", "not supported"),
        True,
        "This mlx-lm version doesn't support this model yet.",
        "Use another model, or llama.cpp with `--engine llamacpp`.",
    ),
    (
        ("no module named",),
        False,
        "The MLX environment is incomplete.",
        "Run `uv cache clean mlx-lm`, then run again so Brasskey sets it up afresh.",
    ),
)  # fmt: skip


def uv_base_command(uv: str, *, offline: bool = False) -> tuple[str, ...]:
    """``uv tool run`` with the pinned mlx-lm environment (the command to run comes after)."""
    return (
        uv, "tool", "run", *(("--offline",) if offline else ()),
        "--managed-python", "--python", UV_PYTHON,
        "--exclude-newer", RESOLVE_CUTOFF,
        "--from", f"mlx-lm=={MLX_LM_VERSION}",
    )  # fmt: skip


def parse_probe(text: str) -> tuple[str, tuple[EngineDevice, ...]] | None:
    """(mlx-lm version, Metal device) from the probe's JSON line; None if unreadable."""
    for line in reversed(text.strip().splitlines()):
        try:
            data: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        payload = cast(dict[str, object], data)
        if "mlx_lm" not in payload:
            continue
        raw_device = payload.get("device")
        device = cast(dict[str, object], raw_device) if isinstance(raw_device, dict) else {}
        limit = device.get("max_recommended_working_set_size")
        name = device.get("device_name") or "Apple GPU"
        devices = (
            (EngineDevice(backend="metal", name=str(name), memory_total_bytes=limit),)
            if isinstance(limit, int)
            else ()
        )
        return f"{payload['mlx_lm']} (mlx {payload.get('mlx', '?')})", devices
    return None


def _interpreter(script: str) -> str | None:
    """The Python that runs a console script (its absolute shebang)."""
    try:
        first = Path(script).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if first.startswith("#!") and first[2:].strip().startswith("/"):
        candidate = first[2:].strip().split()[0]
        if Path(candidate).is_file() and "env" not in Path(candidate).name:
            return candidate
    return None


class MlxAdapter(EngineAdapter):
    id = EngineId.MLX
    supports_api_key = False

    def __init__(
        self,
        *,
        run: CommandRunner = run_command,
        which: Callable[[str], str | None] = shutil.which,
        platform_name: str | None = None,
        fallback_dirs: Sequence[Path] = FALLBACK_DIRS,
    ) -> None:
        self._run = run
        self._which = which
        self._platform = platform_name or sys.platform
        self._fallback_dirs = tuple(fallback_dirs)

    def _find(self, name: str) -> str | None:
        found = self._which(name)
        if found:
            return found
        for directory in self._fallback_dirs:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
        return None

    def detect(self) -> EngineInstall | None:
        if self._platform != "darwin":
            return None
        server = self._find(SERVER)
        if server is not None:
            return self._detect_on_path(server)
        uv = self._find("uv")
        if uv is None:
            logger.debug("Neither %s nor uv found", SERVER)
            return None
        return self._detect_with_uv(uv)

    def _detect_on_path(self, server: str) -> EngineInstall | None:
        python = _interpreter(server)
        probed = None
        if python is not None:
            out = self._run([python, "-c", PROBE], timeout_s=60)
            probed = parse_probe(out.stdout)
        help_out = self._run([server, "--help"], timeout_s=60)
        if help_out.returncode != 0:
            logger.debug("%s --help failed (exit %d)", server, help_out.returncode)
            return None
        version, devices = probed or ("unknown version", ())
        install = EngineInstall(
            engine=self.id,
            version=version,
            executable=(server,),
            flags=frozenset(_FLAG.findall(help_out.stdout + help_out.stderr)),
            devices=devices,
        )
        logger.debug("MLX on PATH: %s %s", server, install.version)
        return install

    def _detect_with_uv(self, uv: str) -> EngineInstall:
        executable = (*uv_base_command(uv), SERVER)
        out = self._run([*uv_base_command(uv, offline=True), "python", "-c", PROBE], timeout_s=60)
        probed = parse_probe(out.stdout) if out.returncode == 0 else None
        if probed is None:
            logger.debug("Pinned MLX environment not set up yet (uv exit %d)", out.returncode)
            return EngineInstall(
                engine=self.id,
                version=MLX_LM_VERSION,
                runtime="uvx",
                executable=executable,
                flags=PINNED_FLAGS,
                needs_setup=True,
                setup_argv=(*executable, "--help"),
                setup_note=SETUP_NOTE,
            )
        version, devices = probed
        logger.debug("Pinned MLX environment ready: %s", version)
        return EngineInstall(
            engine=self.id,
            version=version,
            runtime="uvx",
            executable=executable,
            flags=PINNED_FLAGS,
            devices=devices,
        )

    def install_hint(self, hw: HardwareReport) -> InstallHint:
        if hw.os.family != "darwin":
            return InstallHint(summary="MLX runs only on Apple Silicon Macs.", docs_url=DOCS_URL)
        return InstallHint(
            summary="Install uv so Brasskey can set MLX up automatically (or install mlx-lm).",
            commands=("brew install uv", "or: uv tool install mlx-lm"),
            docs_url=DOCS_URL,
        )

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        if options.api_key_env:
            raise BrasskeyError(
                "MLX's server can't require an API key.",
                fix="Remove serve.api_key_env for MLX, or use `--engine llamacpp`.",
                code="BK-E-ENGINE-API-KEY",
            )
        host = f"[{options.host}]" if ":" in options.host else options.host
        origins = f"http://{host}:{options.port},http://localhost:{options.port}"
        # mlx_lm.server only treats the literal name it was started with as "its" model; any
        # other requested name is loaded as a new model. Starting it with --model <served name>
        # (a link in the engine's private working folder) makes the advertised name work.
        argv: list[str] = [
            *install.executable,
            "--model", options.served_model_name,
            "--host", options.host,
            "--port", str(options.port),
        ]  # fmt: skip
        optional: tuple[tuple[str, ...], ...] = (
            ("--allowed-origins", origins),
            ("--log-level", "INFO"),
            ("--decode-concurrency", str(options.parallel)),
            ("--prompt-concurrency", str(options.parallel)),
        )
        if options.cache_budget_bytes:
            optional += (("--prompt-cache-bytes", str(options.cache_budget_bytes)),)
        for flag in optional:
            if flag[0] in install.flags:
                argv.extend(flag)
        base = f"http://{host}:{options.port}"
        weights = model.artifact.weights_bytes
        return LaunchSpec(
            argv=tuple(argv),
            env=dict(ENGINE_BASE_ENV),
            cwd_links={options.served_model_name: model.snapshot_dir},
            base_url=f"{base}/v1",
            health_url=f"{base}/health",
            startup_timeout_s=max(120.0, weights / 100e6 + 60),
        )

    def explain_failure(
        self, exit_code: int | None, log_tail: Sequence[str]
    ) -> BrasskeyError | None:
        text = "\n".join(log_tail).lower()
        for fragments, require_all, message, fix in _FAILURES:
            check = all if require_all else any
            if check(fragment in text for fragment in fragments):
                return BrasskeyError(message, fix=fix, code="BK-E-ENGINE-START")
        return None
