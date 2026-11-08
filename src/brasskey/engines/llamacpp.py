# SPDX-License-Identifier: Apache-2.0
"""llama.cpp adapter: runs ``llama-server`` (MIT) as a separate process.

Built against llama.cpp 0.9.0 (build 12110) and written to survive version changes:
- ``detect`` reads ``--version``, the flags ``--help`` lists, and the devices ``--list-devices``
  reports (Metal/CUDA/... with their memory: a measurement from the engine itself);
- ``build_launch`` sets every sizing option explicitly (context, GPU layers, slots) and adds
  optional flags only when this version supports them;
- the API key reaches llama.cpp through ``LLAMA_API_KEY`` in its environment, never argv.

Defaults chosen for a private, reproducible server: ``--offline`` (no model downloads by the
engine), ``--no-webui`` (API only), ``--fit off`` (Brasskey sized it; don't silently change it),
``--metrics`` (Prometheus endpoint on the same local port).
"""

import logging
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

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
from brasskey.engines.capabilities import device_pool
from brasskey.engines.types import DevicePool, EngineId
from brasskey.errors import ISSUES_URL, BrasskeyError
from brasskey.hardware.commands import CommandRunner, run_command
from brasskey.hardware.types import MIB, AcceleratorKind, HardwareReport

logger = logging.getLogger(__name__)

EXECUTABLE = "llama-server"
# Where package managers put it when it isn't on PATH (e.g. a GUI-launched shell).
FALLBACK_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path.home() / ".local" / "bin")
DOCS_URL = "https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"
ALL_GPU_LAYERS = "999"  # "offload every layer"; accepted by old and new versions alike

_VERSION = re.compile(r"version:\s*(.+)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")
_DEVICE = re.compile(
    r"^\s*([A-Za-z]+)\d*:\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", re.MULTILINE
)
_BACKENDS = {
    "mtl": "metal",
    "metal": "metal",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "rocm": "rocm",
    "hip": "rocm",
    "sycl": "sycl",
}

# Log fragments (lower case) -> plain-language errors. Checked in order.
_FAILURES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        (
            "failed to allocate",
            "out of memory",
            "cudamalloc failed",
            "unable to allocate",
            "insufficient memory",
        ),
        "llama.cpp ran out of memory while loading the model.",
        "Use a shorter --context or a smaller model (`brasskey run --dry-run` shows what fits), "
        "and close other apps.",
    ),
    (
        ("address already in use", "couldn't bind", "failed to bind"),
        "The server port is already in use.",
        "Pass --port with a free port, or stop the other server.",
    ),
    (
        ("unknown model architecture", "unknown architecture"),
        "This llama.cpp version doesn't support this model yet.",
        "Update llama.cpp (`brew upgrade llama.cpp`), then run again.",
    ),
    (
        ("invalid argument", "unknown argument", "error while handling argument"),
        "llama.cpp rejected an option Brasskey passed (probably a version difference).",
        f"Update llama.cpp (`brew upgrade llama.cpp`); if it persists, report it at {ISSUES_URL}.",
    ),
    (
        ("failed to load model", "error loading model", "gguf_init_from_file"),
        "llama.cpp couldn't load the model file.",
        "Run again (Brasskey re-checks the download); if it persists, update llama.cpp.",
    ),
)


def parse_devices(text: str) -> tuple[EngineDevice, ...]:
    """Devices from ``llama-server --list-devices`` (CPU BLAS entries are skipped)."""
    devices: list[EngineDevice] = []
    for match in _DEVICE.finditer(text):
        backend = _BACKENDS.get(match.group(1).lower())
        if backend is None:
            continue
        devices.append(
            EngineDevice(
                backend=backend,
                name=match.group(2),
                memory_total_bytes=int(match.group(3)) * MIB,
                memory_free_bytes=int(match.group(4)) * MIB,
            )
        )
    return tuple(devices)


def _host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host else host


class LlamaCppAdapter(EngineAdapter):
    id = EngineId.LLAMACPP

    def __init__(self, *, run: CommandRunner = run_command, which: str | None = None) -> None:
        self._run = run
        self._which = which

    def _find_executable(self) -> str | None:
        if self._which is not None:
            return self._which
        found = shutil.which(EXECUTABLE)
        if found:
            return found
        for directory in FALLBACK_DIRS:
            candidate = directory / EXECUTABLE
            if candidate.is_file():
                return str(candidate)
        return None

    def detect(self) -> EngineInstall | None:
        executable = self._find_executable()
        if executable is None:
            logger.debug("llama-server not found on PATH or in %s", FALLBACK_DIRS)
            return None
        version_out = self._run([executable, "--version"], timeout_s=10)
        text = version_out.stdout + version_out.stderr
        match = _VERSION.search(text)
        if not match:
            logger.debug("llama-server --version gave no version (exit %d)", version_out.returncode)
            return None
        help_out = self._run([executable, "--help"], timeout_s=10)
        flags = frozenset(_FLAG.findall(help_out.stdout + help_out.stderr))
        devices: tuple[EngineDevice, ...] = ()
        if "--list-devices" in flags:
            listed = self._run([executable, "--list-devices"], timeout_s=20)
            devices = parse_devices(listed.stdout + listed.stderr)
        install = EngineInstall(
            engine=self.id,
            version=match.group(1).strip(),
            executable=(executable,),
            flags=flags,
            devices=devices,
        )
        logger.debug(
            "llama.cpp %s at %s, %d flags, backends=%s",
            install.version,
            executable,
            len(flags),
            install.backends or ("cpu",),
        )
        return install

    def install_hint(self, hw: HardwareReport) -> InstallHint:
        if hw.os.family == "windows":
            return InstallHint(
                summary="Install llama.cpp with winget.",
                commands=("winget install llama.cpp",),
                docs_url=DOCS_URL,
            )
        if hw.accelerators_of(AcceleratorKind.NVIDIA_GPU):
            return InstallHint(
                summary="Install a CUDA build of llama.cpp so it can use your NVIDIA GPU.",
                commands=(
                    "docker pull ghcr.io/ggml-org/llama.cpp:server-cuda",
                    "or build it: cmake -B build -DGGML_CUDA=ON && cmake --build build -j",
                ),
                docs_url=DOCS_URL,
            )
        return InstallHint(
            summary="Install llama.cpp with Homebrew.",
            commands=("brew install llama.cpp",),
            docs_url=DOCS_URL,
        )

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        pool = device_pool(self.id, hw)
        argv: list[str] = [
            *install.executable,
            "--model",
            str(model.primary_file),
            "--host",
            options.host,
            "--port",
            str(options.port),
            "--ctx-size",
            str(options.context_length * options.parallel),
            "--parallel",
            str(options.parallel),
            "--alias",
            options.served_model_name,
            "--n-gpu-layers",
            "0" if pool == DevicePool.CPU else ALL_GPU_LAYERS,
        ]
        optional: tuple[tuple[str, ...], ...] = (
            ("--jinja",),
            ("--metrics",),
            ("--no-webui",),
            ("--offline",),
            ("--fit", "off"),
        )
        for flag in optional:
            if flag[0] in install.flags:
                argv.extend(flag)
        secret_env = {"LLAMA_API_KEY": options.api_key_env} if options.api_key_env else {}
        host = _host_for_url(options.host)
        base = f"http://{host}:{options.port}"
        weights = model.artifact.weights_bytes
        return LaunchSpec(
            argv=tuple(argv),
            env=dict(ENGINE_BASE_ENV),
            secret_env=secret_env,
            base_url=f"{base}/v1",
            health_url=f"{base}/health",
            metrics_url=f"{base}/metrics" if "--metrics" in install.flags else None,
            # Loading reads the whole file: allow ~100 MB/s plus a fixed margin.
            startup_timeout_s=max(120.0, weights / 100e6 + 60),
        )

    def explain_failure(
        self, exit_code: int | None, log_tail: Sequence[str]
    ) -> BrasskeyError | None:
        text = "\n".join(log_tail).lower()
        for fragments, message, fix in _FAILURES:
            if any(fragment in text for fragment in fragments):
                return BrasskeyError(message, fix=fix, code="BK-E-ENGINE-START")
        return None
