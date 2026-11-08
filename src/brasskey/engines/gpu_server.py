# SPDX-License-Identifier: Apache-2.0
"""What vLLM and SGLang adapters share: Linux + NVIDIA detection, two runtimes, GPU selection.

Detection order (ADR 0010):
1. the engine's console script on PATH (the user installed it, e.g. ``uv tool install vllm``):
   run as a process, with versions read from its own interpreter;
2. otherwise Docker with the NVIDIA Container Toolkit: the engine's official image, pinned by
   digest, pulled once after the user confirms.
On other systems, or Linux without an NVIDIA driver, detection is skipped (no Docker calls).
"""

import logging
import shutil
import sys
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from brasskey.engines.base import ENGINE_BASE_ENV, EngineAdapter, EngineInstall, LocalModel
from brasskey.engines.capabilities import parse_cuda_version
from brasskey.engines.docker import PinnedImage, detect_docker_install
from brasskey.engines.pyenv import probe_package, script_interpreter
from brasskey.hardware.commands import CommandRunner, run_command
from brasskey.hardware.types import AcceleratorKind, HardwareReport

logger = logging.getLogger(__name__)

FALLBACK_DIRS = (Path.home() / ".local" / "bin", Path("/usr/local/bin"))
DRIVER_MARKERS = (Path("/proc/driver/nvidia/version"), Path("/dev/dxg"))  # Linux, WSL2
BF16_MIN_CC = (8, 0)


def nvidia_driver_present() -> bool:
    return any(path.exists() for path in DRIVER_MARKERS) or shutil.which("nvidia-smi") is not None


def wants_float16(model: LocalModel, hw: HardwareReport, gpu_indices: Sequence[int]) -> bool:
    """bf16 weights on a GPU without bfloat16 (before Ampere) must run as float16."""
    if model.artifact.quantization != "bf16":
        return False
    gpus = [
        gpu
        for gpu in hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
        if not gpu_indices or gpu.index in gpu_indices
    ]
    return any((gpu.compute_capability or (0, 0)) < BF16_MIN_CC for gpu in gpus)


def driver_older_than(cuda_version: str | None, hw: HardwareReport) -> bool:
    gpus = hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
    needed = parse_cuda_version(cuda_version)
    supported = parse_cuda_version(gpus[0].cuda_driver_version) if gpus else None
    return needed is not None and supported is not None and supported < needed


def process_env(gpu_indices: Sequence[int]) -> dict[str, str]:
    """Engine environment for the process runtime: no telemetry, and only the chosen GPUs.

    PCI bus order makes CUDA number GPUs like NVML (and ``nvidia-smi``) do.
    """
    env = dict(ENGINE_BASE_ENV)
    if gpu_indices:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    return env


def startup_timeout(weights_bytes: int) -> float:
    """Loading, compiling and capturing CUDA graphs: minutes, more for big models."""
    return max(900.0, weights_bytes / 100e6 + 600.0)


class GpuServerAdapter(EngineAdapter):
    script: ClassVar[str]  # console script that marks a process install, e.g. "vllm"
    package: ClassVar[str]  # distribution name for the version probe
    image: ClassVar[PinnedImage]

    def __init__(
        self,
        *,
        run: CommandRunner = run_command,
        which: Callable[[str], str | None] = shutil.which,
        platform_name: str | None = None,
        has_nvidia_driver: Callable[[], bool] = nvidia_driver_present,
        fallback_dirs: Sequence[Path] = FALLBACK_DIRS,
    ) -> None:
        self._run = run
        self._which = which
        self._platform = platform_name or sys.platform
        self._has_driver = has_nvidia_driver
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
        if not self._platform.startswith("linux"):
            return None
        if not self._has_driver():
            logger.debug("No NVIDIA driver found; skipping %s detection", self.id.display_name)
            return None
        script = self._find(self.script)
        if script is not None:
            return self._detect_process(script)
        return detect_docker_install(self.id, self.image, run=self._run, which=self._which)

    def _detect_process(self, script: str) -> EngineInstall:
        python = script_interpreter(script)
        probed = probe_package(self._run, python, self.package) if python else None
        version, cuda = probed or ("unknown version", None)
        logger.debug("%s on PATH: %s %s (CUDA %s)", self.id.display_name, script, version, cuda)
        return EngineInstall(
            engine=self.id,
            version=version,
            runtime="process",
            executable=self.process_executable(script, python),
            cuda_version=cuda,
        )

    @abstractmethod
    def process_executable(self, script: str, python: str | None) -> tuple[str, ...]:
        """The command that starts the server, before its arguments."""
