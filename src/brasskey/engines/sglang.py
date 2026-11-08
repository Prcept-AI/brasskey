# SPDX-License-Identifier: Apache-2.0
"""SGLang adapter: runs ``sglang.launch_server`` (Apache-2.0) as a process or from its image.

Pinned and checked against SGLang 0.5.22 (released 2026-10-19; CUDA 13.0 builds):
- process runtime: the ``sglang`` console script on PATH; Brasskey starts
  ``<its python> -m sglang.launch_server`` (or ``sglang serve`` if the interpreter is unknown);
- Docker runtime: ``lmsysorg/sglang:v0.5.22-runtime`` by digest (the production image, smaller
  than the development one). Its own entrypoint is kept: it sets up NVIDIA's forward-compatible
  CUDA libraries on older datacenter drivers.

Launch defaults:
- explicit ``--context-length`` and ``--tp-size`` from the plan; SGLang sizes its static memory
  share itself (``--mem-fraction-static`` depends on the GPU and CUDA-graph needs), so Brasskey's
  85% figure stays a planning estimate;
- ``--enable-metrics`` for Prometheus; request logging (``--log-requests``) stays off;
- no API key on the engine: SGLang accepts one only as a command-line argument, which other users
  could read with ``ps``. The engine listens on loopback only and the Brasskey proxy enforces
  keys, as for MLX (ADR 0009).
"""

import logging
from collections.abc import Sequence

from brasskey.engines.base import (
    ENGINE_BASE_ENV,
    EngineInstall,
    InstallHint,
    LaunchSpec,
    LocalModel,
    ServeOptions,
)
from brasskey.engines.capabilities import DRIVER_FOR_CUDA, parse_cuda_version
from brasskey.engines.docker import (
    PinnedImage,
    container_name,
    docker_run_argv,
    model_mount,
    stop_command,
)
from brasskey.engines.gpu_failures import gpu_engine_rules, match_failure
from brasskey.engines.gpu_server import (
    GpuServerAdapter,
    process_env,
    startup_timeout,
    wants_float16,
)
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.types import HardwareReport

logger = logging.getLogger(__name__)

SGLANG_VERSION = "0.5.22"
IMAGE = PinnedImage(
    repository="lmsysorg/sglang",
    tag="v0.5.22-runtime",
    digest="sha256:c3d99f078c329a5b2d0c0c02c4237cf8b0dcd1dc6f875a943ae163c1720854cb",
    engine_version=SGLANG_VERSION,
    cuda_version="13.0",
    download_bytes=14_412_299_832,
    container_port=30000,
)
LAUNCH_MODULE = ("-m", "sglang.launch_server")
DOCS_URL = "https://docs.sglang.io/docs/get-started/install"


class SglangAdapter(GpuServerAdapter):
    id = EngineId.SGLANG
    supports_api_key = False  # only as a command-line argument; the proxy enforces keys
    script = "sglang"
    package = "sglang"
    image = IMAGE

    def process_executable(self, script: str, python: str | None) -> tuple[str, ...]:
        return (python, *LAUNCH_MODULE) if python else (script, "serve")

    def install_hint(self, hw: HardwareReport) -> InstallHint:
        if hw.os.family != "linux":
            return InstallHint(
                summary="SGLang runs on Linux (or WSL2) with an NVIDIA GPU.", docs_url=DOCS_URL
            )
        return InstallHint(
            summary="Install SGLang, or Docker with the NVIDIA Container Toolkit (Brasskey then "
            f"pulls the pinned SGLang {SGLANG_VERSION} image).",
            commands=(
                f"uv tool install sglang=={SGLANG_VERSION}",
                "or Docker + NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/"
                "cloud-native/container-toolkit/latest/install-guide.html",
            ),
            docs_url=DOCS_URL,
        )

    def _server_args(
        self, model_path: str, host: str, port: int, options: ServeOptions, half: bool
    ) -> list[str]:
        args = [
            "--model-path", model_path,
            "--host", host,
            "--port", str(port),
            "--served-model-name", options.served_model_name,
            "--context-length", str(options.context_length),
            "--tp-size", str(options.tensor_parallel),
            "--enable-metrics",
        ]  # fmt: skip
        if half:
            args += ["--dtype", "half"]
        return args

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        if options.api_key_env:
            raise BrasskeyError(
                "SGLang's server can take an API key only on its command line.",
                fix="Let the Brasskey proxy enforce the key (the default), or use --engine vllm.",
                code="BK-E-ENGINE-API-KEY",
            )
        half = wants_float16(model, hw, options.gpu_indices)
        if install.runtime == "docker":
            mount = model_mount(model.snapshot_dir)
            name = container_name(self.id, options.port)
            # Inside the container SGLang listens on all of *its* interfaces so Docker's port
            # mapping reaches it; on the host the port is published on 127.0.0.1 only.
            inside = "0.0.0.0"  # noqa: S104
            argv = docker_run_argv(
                install,
                self.image,
                name=name,
                host_port=options.port,
                gpu_indices=options.gpu_indices,
                mount=mount,
                env=dict(ENGINE_BASE_ENV),
                command=(
                    "python3",
                    *LAUNCH_MODULE,
                    *self._server_args(
                        mount.model_path, inside, self.image.container_port, options, half
                    ),
                ),
            )
            env: dict[str, str] = {}
            stop: tuple[str, ...] = stop_command(install, name)
            placeholders = {str(mount.host_dir): "{model_cache}"}
        else:
            argv = [
                *install.executable,
                *self._server_args(
                    str(model.snapshot_dir), "127.0.0.1", options.port, options, half
                ),
            ]
            env = process_env(options.gpu_indices)
            stop = ()
            placeholders = {str(model.snapshot_dir): "{model}"}
        base = f"http://127.0.0.1:{options.port}"
        logger.debug(
            "SGLang launch: runtime=%s tp=%d gpus=%s half=%s",
            install.runtime, options.tensor_parallel, options.gpu_indices, half,
        )  # fmt: skip
        return LaunchSpec(
            argv=tuple(argv),
            env=env,
            base_url=f"{base}/v1",
            health_url=f"{base}/health",  # 503 while starting and warming up, then 200
            metrics_url=f"{base}/metrics",
            startup_timeout_s=startup_timeout(model.artifact.weights_bytes),
            stop_argv=stop,
            placeholders=placeholders,
        )

    def explain_failure(
        self, exit_code: int | None, log_tail: Sequence[str]
    ) -> BrasskeyError | None:
        cuda = parse_cuda_version(IMAGE.cuda_version) or (13, 0)
        rules = gpu_engine_rules(
            "SGLang", driver_for_cuda=DRIVER_FOR_CUDA.get(cuda, "a newer"), image=IMAGE.ref
        )
        return match_failure(rules, log_tail)
