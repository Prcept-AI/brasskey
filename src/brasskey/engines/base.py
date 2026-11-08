# SPDX-License-Identifier: Apache-2.0
"""The engine adapter contract (ADR 0005).

Every engine (llama.cpp, MLX, vLLM and SGLang) implements :class:`EngineAdapter`:

- ``detect``         is it installed? which version, which flags, which devices?
- ``install_hint``   the exact command to install it when it isn't
- ``build_launch``   the command line and environment to start it (never secrets in argv)
- ``explain_failure``  turn the engine's log into a plain-language error with a fix
- ``health``         ready / starting / failed, from the engine's health endpoint

Starting, stopping and watching the process is shared (``brasskey.runtime.supervisor``), so an
adapter is mostly pure functions and is easy to test with golden command lines. What engines can
run on which hardware lives in ``brasskey.engines.capabilities`` (milestone c) and is reused here.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field

from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.types import HardwareReport
from brasskey.models.types import ModelArtifact

# Brasskey's environment variable for the per-run secret the proxy uses to reach the engine.
INTERNAL_KEY_ENV = "BRASSKEY_ENGINE_KEY"

# Set for every engine process: no telemetry, no surprise downloads (Brasskey downloads first).
ENGINE_BASE_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EngineDevice(_Frozen):
    """A compute device as the engine itself reports it (a measurement, not an estimate)."""

    backend: str  # e.g. "metal", "cuda", "cpu"
    name: str
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None


class EngineInstall(_Frozen):
    engine: EngineId
    version: str
    runtime: Literal["process", "uvx", "docker"] = "process"
    executable: tuple[str, ...] = Field(min_length=1)
    flags: frozenset[str] = Field(
        default=frozenset(), description="Command-line flags this version supports."
    )
    devices: tuple[EngineDevice, ...] = ()
    devices_listed: bool = Field(
        default=False,
        description="The engine listed its devices itself, so `devices` is complete: no GPU "
        "device means it can use only the CPU. False: unknown (e.g. an older version).",
    )
    image_digest: str | None = None
    image: str | None = Field(
        default=None, description="Docker runtime: the pinned image, repository:tag@sha256:..."
    )
    cuda_version: str | None = Field(
        default=None, description='CUDA version the engine was built with, e.g. "13.0".'
    )
    needs_setup: bool = Field(
        default=False, description="A one-time setup (e.g. a pinned uv environment) must run first."
    )
    setup_argv: tuple[str, ...] = ()
    setup_note: str | None = Field(default=None, description='e.g. "about 70 MB, one time".')
    setup_timeout_s: float = Field(
        default=1800.0, gt=0, description="Longest the one-time setup may take (a big image pull)."
    )

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(sorted({device.backend for device in self.devices}))

    @property
    def gpu_backends(self) -> frozenset[str] | None:
        """GPU backends the engine reported itself (empty: CPU only); None when unknown."""
        return frozenset(self.backends) if self.devices_listed else None


class InstallHint(_Frozen):
    summary: str
    commands: tuple[str, ...] = ()
    docs_url: str | None = None


class ServeOptions(_Frozen):
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    served_model_name: str
    context_length: int = Field(gt=0)
    parallel: int = Field(default=1, ge=1, description="Concurrent sequences (server slots).")
    api_key_env: str | None = Field(
        default=None,
        description="NAME of the environment variable holding the API key; the key itself never "
        "enters argv, logs or lock files.",
    )
    cache_budget_bytes: int | None = Field(
        default=None,
        ge=0,
        description="KV-cache memory the plan allows, for engines that size caches themselves.",
    )
    tensor_parallel: int = Field(
        default=1, ge=1, description="GPUs one model is split across (vLLM, SGLang)."
    )
    gpu_indices: tuple[int, ...] = Field(
        default=(), description="NVML indices of the GPUs to use (empty: the engine's default)."
    )


class LocalModel(_Frozen):
    """An artifact downloaded and verified on disk."""

    artifact: ModelArtifact
    snapshot_dir: Path
    files: tuple[Path, ...]

    @property
    def primary_file(self) -> Path:
        """The file an engine opens (first GGUF part, or the snapshot folder for safetensors)."""
        weights = sorted(path for path in self.files if path.suffix == ".gguf")
        return weights[0] if weights else self.snapshot_dir


class LaunchSpec(_Frozen):
    argv: tuple[str, ...] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict, description="Non-secret additions.")
    secret_env: dict[str, str] = Field(
        default_factory=dict,
        description="Engine env var -> Brasskey env var holding the secret (copied at start).",
    )
    base_url: str
    health_url: str
    metrics_url: str | None = None
    startup_timeout_s: float = Field(default=600.0, gt=0)
    cwd_links: dict[str, Path] = Field(
        default_factory=dict,
        description="Links (relative name -> target) the supervisor creates in a private working "
        "folder that becomes the engine's working directory.",
    )
    stop_argv: tuple[str, ...] = Field(
        default=(),
        description="A command that stops the server cleanly (e.g. `docker stop <name>`), run "
        "before signals; signals alone only reach the Docker client, not the container.",
    )
    placeholders: dict[str, str] = Field(
        default_factory=dict,
        description="Machine-specific argument text -> placeholder for brasskey.lock (no absolute "
        "paths in locks).",
    )


class HealthStatus(StrEnum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


class EngineStartError(BrasskeyError):
    default_code = "BK-E-ENGINE-START"


class EngineNotInstalledError(BrasskeyError):
    default_code = "BK-E-ENGINE-MISSING"


class EngineAdapter(ABC):
    id: ClassVar[EngineId]
    supports_api_key: ClassVar[bool] = True  # can the engine's server require an API key?

    @abstractmethod
    def detect(self) -> EngineInstall | None:
        """The installed engine, or None. Read-only and quick (a few subprocess calls)."""

    @abstractmethod
    def install_hint(self, hw: HardwareReport) -> InstallHint: ...

    @abstractmethod
    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec: ...

    def explain_failure(
        self, exit_code: int | None, log_tail: Sequence[str]
    ) -> BrasskeyError | None:
        """A plain-language error for a failed start, or None to show the log tail."""
        return None

    def health(self, spec: LaunchSpec, client: httpx2.Client) -> HealthStatus:
        """READY on HTTP 200 from the health URL, STARTING while it refuses or answers 503."""
        try:
            response = client.get(spec.health_url)
        except httpx2.HTTPError:
            return HealthStatus.STARTING
        if response.status_code == 200:
            return HealthStatus.READY
        if response.status_code in (502, 503, 504):
            return HealthStatus.STARTING
        return HealthStatus.FAILED
