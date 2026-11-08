# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter: runs ``vllm serve`` (Apache-2.0) as a process or from its official image.

Pinned and checked against vLLM 0.32.0 (released 2026-10-18; CUDA 13.0 builds):
- process runtime: ``vllm`` on PATH, e.g. ``uv tool install vllm==0.32.0``;
- Docker runtime: ``vllm/vllm-openai:v0.32.0`` by digest, run as the image's non-root user
  (UID 2000), with ``VLLM_ENABLE_CUDA_COMPATIBILITY=1`` when the driver predates CUDA 13.0.

Launch defaults for a private, reproducible server:
- sizing is explicit: ``--max-model-len`` (the planned context), ``--gpu-memory-utilization``
  (the planned share of each GPU), ``--tensor-parallel-size`` (the planned GPU count);
- prompts are never logged: request logging is off by default in vLLM 0.32 and Brasskey never
  turns it on; the uvicorn access log is off too;
- the API key reaches vLLM through ``VLLM_API_KEY`` (its documented variable), never argv. vLLM
  guards only ``/v1``-style paths with it, which is why the engine port stays on loopback behind
  the Brasskey proxy;
- bf16 weights on GPUs without bfloat16 (T4, compute capability 7.5) run as float16.
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
from brasskey.engines.capabilities import DRIVER_FOR_CUDA, memory_profile, parse_cuda_version
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
    driver_older_than,
    process_env,
    startup_timeout,
    wants_float16,
)
from brasskey.engines.types import DevicePool, EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.types import HardwareReport

logger = logging.getLogger(__name__)

VLLM_VERSION = "0.32.0"
IMAGE = PinnedImage(
    repository="vllm/vllm-openai",
    tag="v0.32.0",
    digest="sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34",
    engine_version=VLLM_VERSION,
    cuda_version="13.0",
    download_bytes=9_205_727_079,
    container_port=8000,
)
CONTAINER_USER = "2000:0"  # the image's documented non-root user (vllm, group 0)
KEY_ENV = "VLLM_API_KEY"
DOCS_URL = "https://docs.vllm.ai/en/stable/getting_started/installation/gpu/"


class VllmAdapter(GpuServerAdapter):
    id = EngineId.VLLM
    script = "vllm"
    package = "vllm"
    image = IMAGE

    def process_executable(self, script: str, python: str | None) -> tuple[str, ...]:
        return (script, "serve")

    def install_hint(self, hw: HardwareReport) -> InstallHint:
        if hw.os.family != "linux":
            return InstallHint(
                summary="vLLM runs on Linux (or WSL2) with an NVIDIA GPU.", docs_url=DOCS_URL
            )
        return InstallHint(
            summary="Install vLLM, or Docker with the NVIDIA Container Toolkit (Brasskey then "
            f"pulls the pinned vLLM {VLLM_VERSION} image).",
            commands=(
                f"uv tool install vllm=={VLLM_VERSION}",
                "or Docker + NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/"
                "cloud-native/container-toolkit/latest/install-guide.html",
            ),
            docs_url=DOCS_URL,
        )

    def _serve_args(
        self, model_path: str, host: str, port: int, options: ServeOptions, half: bool
    ) -> list[str]:
        utilization = memory_profile(self.id, DevicePool.NVIDIA_GPU).max_fraction
        args = [
            model_path,
            "--host", host,
            "--port", str(port),
            "--served-model-name", options.served_model_name,
            "--max-model-len", str(options.context_length),
            "--gpu-memory-utilization", f"{utilization:.2f}",
            "--tensor-parallel-size", str(options.tensor_parallel),
            "--disable-uvicorn-access-log",
        ]  # fmt: skip
        if half:
            args += ["--dtype", "half"]
        return args

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        half = wants_float16(model, hw, options.gpu_indices)
        secret_env = {KEY_ENV: options.api_key_env} if options.api_key_env else {}
        if install.runtime == "docker":
            mount = model_mount(model.snapshot_dir)
            name = container_name(self.id, options.port)
            container_env = dict(ENGINE_BASE_ENV)
            if driver_older_than(install.cuda_version, hw):
                container_env["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"  # datacenter forward compat
            # Inside the container vLLM listens on all of *its* interfaces so Docker's port
            # mapping reaches it; on the host the port is published on 127.0.0.1 only.
            inside = "0.0.0.0"  # noqa: S104
            argv = docker_run_argv(
                install,
                self.image,
                name=name,
                host_port=options.port,
                gpu_indices=options.gpu_indices,
                mount=mount,
                env=container_env,
                pass_env=tuple(secret_env),
                user=CONTAINER_USER,
                command=self._serve_args(
                    mount.model_path, inside, self.image.container_port, options, half
                ),
            )
            env: dict[str, str] = {}
            stop: tuple[str, ...] = stop_command(install, name)
            placeholders = {str(mount.host_dir): "{model_cache}"}
        else:
            argv = [
                *install.executable,
                *self._serve_args(
                    str(model.snapshot_dir), "127.0.0.1", options.port, options, half
                ),
            ]
            env = process_env(options.gpu_indices)
            stop = ()
            placeholders = {str(model.snapshot_dir): "{model}"}
        base = f"http://127.0.0.1:{options.port}"
        logger.debug(
            "vLLM launch: runtime=%s tp=%d gpus=%s half=%s",
            install.runtime, options.tensor_parallel, options.gpu_indices, half,
        )  # fmt: skip
        return LaunchSpec(
            argv=tuple(argv),
            env=env,
            secret_env=secret_env,
            base_url=f"{base}/v1",
            health_url=f"{base}/health",
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
            "vLLM", driver_for_cuda=DRIVER_FOR_CUDA.get(cuda, "a newer"), image=IMAGE.ref
        )
        return match_failure(rules, log_tail)
