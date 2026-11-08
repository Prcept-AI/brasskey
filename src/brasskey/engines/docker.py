# SPDX-License-Identifier: Apache-2.0
"""The Docker runtime shared by engines with official images (vLLM, SGLang).

Brasskey never builds or modifies images: it runs the engine's own image, pinned by digest, and
only after the user confirmed the one-time pull. Every container it starts:

- publishes the engine port on ``127.0.0.1`` only (the Brasskey proxy is the public face);
- mounts the model's Hugging Face cache folder read-only and runs with ``--pull never`` (no
  surprise downloads at start);
- gets only the GPUs the plan chose (``--gpus "device=..."``), no new privileges, a label, and a
  name derived from its private port, so ``docker stop`` can end it cleanly;
- receives secrets by variable *name* (``-e VLLM_API_KEY``): Docker copies the value from the
  client's environment, so it never appears on a command line.
"""

import json
import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from brasskey.engines.base import EngineInstall
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandRunner

logger = logging.getLogger(__name__)

DOCKER = "docker"
# Present when the NVIDIA Container Toolkit is installed (it makes `--gpus` work).
NVIDIA_TOOLKIT_BINARIES = ("nvidia-container-runtime-hook", "nvidia-ctk", "nvidia-container-cli")
TOOLKIT_DOCS = (
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
)
SHM_SIZE = "16g"  # shared memory for multi-GPU (NCCL) and engine IPC; a limit, not a reservation
STOP_TIMEOUT_S = 20
PULL_TIMEOUT_S = 3 * 3600.0  # 10-15 GB images on slow links
LABEL = "brasskey.managed=true"


@dataclass(frozen=True)
class PinnedImage:
    """An engine image Brasskey has checked, pinned by its multi-architecture digest."""

    repository: str
    tag: str
    digest: str
    engine_version: str
    cuda_version: str
    download_bytes: int  # compressed linux/amd64 size reported by Docker Hub
    container_port: int

    @property
    def ref(self) -> str:
        return f"{self.repository}:{self.tag}@{self.digest}"

    @property
    def setup_note(self) -> str:
        return f"a {self.download_bytes / 1e9:.1f} GB Docker image, one time"


@dataclass(frozen=True)
class DockerHost:
    executable: str
    server_version: str
    nvidia_toolkit: bool


@dataclass(frozen=True)
class ModelMount:
    host_dir: Path
    container_dir: str
    model_path: str  # what the engine opens inside the container


def parse_docker_info(text: str) -> tuple[str, frozenset[str]] | None:
    """(server version, runtime names) from ``docker info --format '{{json .}}'``."""
    try:
        data: object = json.loads(text.strip().splitlines()[-1]) if text.strip() else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    info = cast(dict[str, object], data)
    version = info.get("ServerVersion")
    if not isinstance(version, str) or not version:
        return None  # the client works but the daemon didn't answer
    runtimes = info.get("Runtimes")
    names: frozenset[str] = frozenset()
    if isinstance(runtimes, dict):
        names = frozenset(cast(dict[str, object], runtimes))
    return version, names


def docker_host(run: CommandRunner, which: Callable[[str], str | None]) -> DockerHost | None:
    """A reachable Docker daemon, or None (not installed, not running, or no permission)."""
    executable = which(DOCKER)
    if executable is None:
        return None
    out = run([executable, "info", "--format", "{{json .}}"], timeout_s=15)
    parsed = parse_docker_info(out.stdout) if out.ok else None
    if parsed is None:
        logger.debug("docker info failed (exit %d): daemon not reachable", out.returncode)
        return None
    version, runtimes = parsed
    toolkit = "nvidia" in runtimes or any(which(name) for name in NVIDIA_TOOLKIT_BINARIES)
    return DockerHost(executable, version, toolkit)


def image_present(run: CommandRunner, docker: str, image: PinnedImage) -> bool:
    out = run([docker, "image", "inspect", "--format", "{{.Id}}", image.ref], timeout_s=15)
    return out.ok


def detect_docker_install(
    engine: EngineId,
    image: PinnedImage,
    *,
    run: CommandRunner,
    which: Callable[[str], str | None] = shutil.which,
) -> EngineInstall | None:
    """The engine as its pinned image, when Docker can use NVIDIA GPUs here."""
    host = docker_host(run, which)
    if host is None:
        return None
    if not host.nvidia_toolkit:
        logger.debug("Docker %s found without the NVIDIA Container Toolkit", host.server_version)
        return None
    present = image_present(run, host.executable, image)
    logger.debug(
        "%s image %s %s", engine.display_name, image.tag, "present" if present else "not pulled"
    )
    return EngineInstall(
        engine=engine,
        version=image.engine_version,
        runtime="docker",
        executable=(host.executable,),
        image=image.ref,
        image_digest=image.digest,
        cuda_version=image.cuda_version,
        needs_setup=not present,
        setup_argv=() if present else (host.executable, "pull", image.ref),
        setup_note=None if present else image.setup_note,
        setup_timeout_s=PULL_TIMEOUT_S,
    )


def model_mount(snapshot_dir: Path) -> ModelMount:
    """Mount the model's Hugging Face cache folder read-only.

    Files in ``snapshots/<revision>/`` are relative links into ``../../blobs/``, so the whole repo
    folder is mounted and the engine opens the snapshot inside it.
    """
    if snapshot_dir.parent.name == "snapshots":
        repo_dir = snapshot_dir.parent.parent
        container_dir = f"/models/{repo_dir.name}"
        return ModelMount(repo_dir, container_dir, f"{container_dir}/snapshots/{snapshot_dir.name}")
    container_dir = f"/models/{snapshot_dir.name}"
    return ModelMount(snapshot_dir, container_dir, container_dir)


def gpus_option(gpu_indices: Sequence[int]) -> str:
    """The ``--gpus`` value: the quotes keep Docker from splitting the device list at commas."""
    if not gpu_indices:
        return "all"
    return '"device=' + ",".join(str(index) for index in gpu_indices) + '"'


def container_name(engine: EngineId, host_port: int) -> str:
    """Unique while the server runs: its private port can't be in use twice."""
    return f"brasskey-{engine.value}-{host_port}"


def docker_run_argv(
    install: EngineInstall,
    image: PinnedImage,
    *,
    name: str,
    host_port: int,
    gpu_indices: Sequence[int],
    mount: ModelMount,
    env: Mapping[str, str],
    pass_env: Sequence[str] = (),
    user: str | None = None,
    command: Sequence[str] = (),
) -> list[str]:
    """``docker run`` for one engine server (attached: its output goes to Brasskey's log)."""
    if ":" in str(mount.host_dir):
        raise BrasskeyError(
            f"Docker can't mount a folder whose path contains ':' ({mount.host_dir}).",
            fix="Set HF_HOME to a folder without ':' in its path, then run again.",
            code="BK-E-DOCKER-MOUNT",
        )
    argv = [
        *install.executable, "run", "--rm", "--pull", "never",
        "--name", name, "--label", LABEL,
        "--gpus", gpus_option(gpu_indices),
        "--shm-size", SHM_SIZE,
        "--security-opt", "no-new-privileges",
        "-p", f"127.0.0.1:{host_port}:{image.container_port}",
        "-v", f"{mount.host_dir}:{mount.container_dir}:ro",
    ]  # fmt: skip
    if user is not None:
        argv += ["--user", user]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    for key in pass_env:
        argv += ["-e", key]  # value copied from the client's environment, never on argv
    return [*argv, install.image or image.ref, *command]


def stop_command(install: EngineInstall, name: str) -> tuple[str, ...]:
    """Stops the container: SIGTERM to the engine, SIGKILL after the timeout."""
    return (*install.executable, "stop", "--time", str(STOP_TIMEOUT_S), name)
