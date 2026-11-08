# SPDX-License-Identifier: Apache-2.0
"""vLLM and SGLang adapters: detection (PATH / Docker), launch goldens, GPU rules, failures."""

import json
import sys
from pathlib import Path

import pytest

from brasskey.engines.base import EngineInstall, LocalModel, ServeOptions
from brasskey.engines.capabilities import (
    check_cuda_driver,
    check_engine,
    choose_gpus,
    parse_cuda_version,
)
from brasskey.engines.docker import (
    gpus_option,
    model_mount,
    parse_docker_info,
)
from brasskey.engines.pyenv import parse_probe, probe_package
from brasskey.engines.sglang import IMAGE as SGLANG_IMAGE
from brasskey.engines.sglang import SglangAdapter
from brasskey.engines.types import EngineId
from brasskey.engines.vllm import IMAGE as VLLM_IMAGE
from brasskey.engines.vllm import VllmAdapter
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandOutput, run_command
from brasskey.hardware.types import AcceleratorKind, HardwareReport
from brasskey.models.catalog import find_model, load_catalog
from fakes import machines

DOCKER = "/usr/bin/docker"
DOCKER_INFO = json.dumps({"ServerVersion": "28.5.1", "Runtimes": {"nvidia": {}, "runc": {}}})
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
INSIDE = "0.0.0.0"  # noqa: S104 (inside the container; published on 127.0.0.1 on the host)


class DockerRunner:
    """Answers `docker info`, `docker image inspect` and the version probe like the real tools."""

    def __init__(self, *, info: str = DOCKER_INFO, pulled: bool = True, probe: str = "") -> None:
        self.info = info
        self.pulled = pulled
        self.probe = probe
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, timeout_s=5.0):  # type: ignore[no-untyped-def]
        self.calls.append(list(argv))
        if argv[1:3] == ["info", "--format"]:
            return CommandOutput(0, self.info) if self.info else CommandOutput(1, "", "denied")
        if argv[1:3] == ["image", "inspect"]:
            return CommandOutput(0 if self.pulled else 1, "sha256:abc" if self.pulled else "")
        if "-c" in argv:
            return CommandOutput(0, self.probe)
        return CommandOutput(127, "", "unexpected")


def adapter(cls, runner, *, tools=None, platform="linux", driver=True):  # type: ignore[no-untyped-def]
    found = {"docker": DOCKER, **(tools or {})}
    return cls(
        run=runner,
        which=found.get,
        platform_name=platform,
        has_nvidia_driver=lambda: driver,
        fallback_dirs=(),
    )


def local_model(tmp_path: Path, quantization: str = "bf16") -> LocalModel:
    model = find_model(load_catalog(), "qwen3-0.6b")
    artifact = next(a for a in model.artifacts_for(EngineId.VLLM) if a.quantization == quantization)
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / REV
    snapshot.mkdir(parents=True)
    return LocalModel(artifact=artifact, snapshot_dir=snapshot, files=(snapshot / "config.json",))


def options(**overrides) -> ServeOptions:  # type: ignore[no-untyped-def]
    values = {"port": 41234, "served_model_name": "qwen3-0.6b", "context_length": 8192}
    return ServeOptions(**(values | overrides))


def docker_install(engine: EngineId, image) -> EngineInstall:  # type: ignore[no-untyped-def]
    return EngineInstall(
        engine=engine, version=image.engine_version, runtime="docker", executable=(DOCKER,),
        image=image.ref, image_digest=image.digest, cuda_version=image.cuda_version,
    )  # fmt: skip


def with_driver(hw: HardwareReport, cuda: str, driver: str) -> HardwareReport:
    gpus = tuple(
        gpu.model_copy(update={"cuda_driver_version": cuda, "driver_version": driver})
        for gpu in hw.accelerators
    )
    return hw.model_copy(update={"accelerators": gpus})


L4 = with_driver(machines.l4(), "13.0", "580.95.05")


# --- detection ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [VllmAdapter, SglangAdapter])
def test_detection_is_skipped_off_linux_or_without_a_driver(cls):
    runner = DockerRunner()
    assert adapter(cls, runner, platform="darwin").detect() is None
    assert adapter(cls, runner, driver=False).detect() is None
    assert runner.calls == []  # no Docker calls on machines that can't run these engines


@pytest.mark.parametrize(
    ("cls", "image"), [(VllmAdapter, VLLM_IMAGE), (SglangAdapter, SGLANG_IMAGE)]
)
def test_docker_with_the_pinned_image(cls, image):
    install = adapter(cls, DockerRunner()).detect()
    assert install is not None
    assert install.runtime == "docker"
    assert install.image == image.ref
    assert "@sha256:" in install.image
    assert install.needs_setup is False
    assert install.cuda_version == "13.0"


def test_image_not_pulled_yet_needs_a_confirmed_pull():
    install = adapter(VllmAdapter, DockerRunner(pulled=False)).detect()
    assert install is not None
    assert install.needs_setup
    assert install.setup_argv == (DOCKER, "pull", VLLM_IMAGE.ref)
    assert install.setup_note == "a 9.2 GB Docker image, one time"
    assert install.setup_timeout_s >= 3600


def test_docker_without_the_nvidia_toolkit_or_daemon_is_not_an_install():
    no_toolkit = json.dumps({"ServerVersion": "28.5.1", "Runtimes": {"runc": {}}})
    assert adapter(VllmAdapter, DockerRunner(info=no_toolkit)).detect() is None
    assert adapter(VllmAdapter, DockerRunner(info="")).detect() is None  # daemon unreachable
    with_hook = adapter(
        VllmAdapter, DockerRunner(info=no_toolkit), tools={"nvidia-ctk": "/usr/bin/nvidia-ctk"}
    ).detect()
    assert with_hook is not None  # CDI-style setups: the toolkit is installed, runtime not listed


def test_vllm_on_path_runs_as_a_process(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    script = tmp_path / "venv" / "bin" / "vllm"
    script.write_text(f"#!{python}\nfrom vllm.entrypoints.cli.main import main\n")
    runner = DockerRunner(probe='{"version": "0.32.0", "cuda": "13.0"}\n')
    install = adapter(VllmAdapter, runner, tools={"vllm": str(script)}).detect()
    assert install is not None
    assert install.runtime == "process"
    assert install.version == "0.32.0"
    assert install.cuda_version == "13.0"
    assert install.executable == (str(script), "serve")
    assert runner.calls[0][:2] == [str(python), "-c"]  # probed with its own interpreter


def test_sglang_on_path_uses_its_interpreter(tmp_path):
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("")
    script = tmp_path / "bin" / "sglang"
    script.write_text(f"#!{python}\n")
    runner = DockerRunner(probe='{"version": "0.5.22", "cuda": "13.0"}')
    install = adapter(SglangAdapter, runner, tools={"sglang": str(script)}).detect()
    assert install is not None
    assert install.executable == (str(python), "-m", "sglang.launch_server")


def test_the_version_probe_really_runs_without_importing_the_package():
    found = probe_package(run_command, sys.executable, "pytest")
    assert found is not None
    assert found[0] == pytest.__version__
    assert parse_probe('noise\n{"version": "1.0"}\n') == ("1.0", None)
    assert parse_probe("garbage") is None


def test_parse_docker_info():
    assert parse_docker_info(DOCKER_INFO) == ("28.5.1", frozenset({"nvidia", "runc"}))
    assert parse_docker_info('{"ServerVersion": ""}') is None  # client only, no daemon
    assert parse_docker_info("not json") is None


# --- launch goldens ----------------------------------------------------------------------------


def test_vllm_process_launch(tmp_path):
    model = local_model(tmp_path)
    install = EngineInstall(
        engine=EngineId.VLLM, version="0.32.0", executable=("/opt/vllm/bin/vllm", "serve"),
        cuda_version="13.0",
    )  # fmt: skip
    spec = VllmAdapter().build_launch(
        install, model, options(api_key_env="BRASSKEY_ENGINE_KEY", gpu_indices=(1,)), L4
    )
    assert spec.argv == (
        "/opt/vllm/bin/vllm", "serve", str(model.snapshot_dir),
        "--host", "127.0.0.1", "--port", "41234",
        "--served-model-name", "qwen3-0.6b", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90", "--tensor-parallel-size", "1",
        "--disable-uvicorn-access-log",
    )  # fmt: skip
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert spec.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert spec.env["VLLM_NO_USAGE_STATS"] == "1"
    assert spec.secret_env == {"VLLM_API_KEY": "BRASSKEY_ENGINE_KEY"}  # by name, never argv
    assert spec.health_url == "http://127.0.0.1:41234/health"
    assert spec.metrics_url == "http://127.0.0.1:41234/metrics"
    assert spec.stop_argv == ()
    assert spec.placeholders == {str(model.snapshot_dir): "{model}"}
    assert "--enable-log-requests" not in spec.argv  # prompts are never logged


def test_vllm_docker_launch(tmp_path):
    model = local_model(tmp_path)
    install = docker_install(EngineId.VLLM, VLLM_IMAGE)
    hw = with_driver(machines.l4(count=4), "13.0", "580.95.05")
    spec = VllmAdapter().build_launch(
        install,
        model,
        options(api_key_env="BRASSKEY_ENGINE_KEY", tensor_parallel=2, gpu_indices=(0, 2)),
        hw,
    )
    repo = model.snapshot_dir.parent.parent
    container = "/models/models--Qwen--Qwen3-0.6B"
    assert spec.argv == (
        DOCKER, "run", "--rm", "--pull", "never",
        "--name", "brasskey-vllm-41234", "--label", "brasskey.managed=true",
        "--gpus", '"device=0,2"', "--shm-size", "16g",
        "--security-opt", "no-new-privileges",
        "-p", "127.0.0.1:41234:8000",
        "-v", f"{repo}:{container}:ro",
        "--user", "2000:0",
        "-e", "DO_NOT_TRACK=1", "-e", "HF_HUB_DISABLE_TELEMETRY=1", "-e", "HF_HUB_OFFLINE=1",
        "-e", "VLLM_NO_USAGE_STATS=1",
        "-e", "VLLM_API_KEY",
        VLLM_IMAGE.ref,
        f"{container}/snapshots/{REV}",
        "--host", INSIDE, "--port", "8000",
        "--served-model-name", "qwen3-0.6b", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90", "--tensor-parallel-size", "2",
        "--disable-uvicorn-access-log",
    )  # fmt: skip
    assert spec.secret_env == {"VLLM_API_KEY": "BRASSKEY_ENGINE_KEY"}
    assert spec.stop_argv == (DOCKER, "stop", "--time", "20", "brasskey-vllm-41234")
    assert spec.placeholders == {str(repo): "{model_cache}"}
    assert spec.base_url == "http://127.0.0.1:41234/v1"


def test_old_driver_turns_on_forward_compatibility_in_the_vllm_image(tmp_path):
    install = docker_install(EngineId.VLLM, VLLM_IMAGE)
    old = with_driver(machines.l4(), "12.8", "570.172.08")
    spec = VllmAdapter().build_launch(install, local_model(tmp_path), options(), old)
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1" in spec.argv


@pytest.mark.parametrize("cls", [VllmAdapter, SglangAdapter])
def test_bf16_on_a_t4_runs_as_float16(cls, tmp_path):
    install = docker_install(cls.id, cls.image)
    t4 = cls().build_launch(install, local_model(tmp_path), options(), machines.t4())
    assert t4.argv[-2:] == ("--dtype", "half")
    l4 = cls().build_launch(install, local_model(tmp_path / "l4"), options(), L4)
    assert "--dtype" not in l4.argv


def test_sglang_docker_launch_keeps_the_image_entrypoint(tmp_path):
    model = local_model(tmp_path)
    install = docker_install(EngineId.SGLANG, SGLANG_IMAGE)
    spec = SglangAdapter().build_launch(install, model, options(gpu_indices=(0,)), L4)
    image_at = spec.argv.index(SGLANG_IMAGE.ref)
    assert "--entrypoint" not in spec.argv  # NVIDIA's entrypoint sets up CUDA compatibility
    assert "--user" not in spec.argv
    assert spec.argv[image_at + 1 :] == (
        "python3", "-m", "sglang.launch_server",
        "--model-path", f"/models/models--Qwen--Qwen3-0.6B/snapshots/{REV}",
        "--host", INSIDE, "--port", "30000",
        "--served-model-name", "qwen3-0.6b", "--context-length", "8192", "--tp-size", "1",
        "--enable-metrics",
    )  # fmt: skip
    assert "-p" in spec.argv
    assert spec.argv[spec.argv.index("-p") + 1] == "127.0.0.1:41234:30000"
    assert spec.secret_env == {}


def test_sglang_process_launch(tmp_path):
    model = local_model(tmp_path)
    install = EngineInstall(
        engine=EngineId.SGLANG, version="0.5.22", executable=("/v/bin/python", "-m",
        "sglang.launch_server"),
    )  # fmt: skip
    spec = SglangAdapter().build_launch(install, model, options(gpu_indices=(0,)), L4)
    assert spec.argv[:5] == ("/v/bin/python", "-m", "sglang.launch_server", "--model-path",
                             str(model.snapshot_dir))  # fmt: skip
    assert "--log-requests" not in spec.argv
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"


def test_sglang_never_takes_a_key_on_its_command_line(tmp_path):
    install = docker_install(EngineId.SGLANG, SGLANG_IMAGE)
    assert SglangAdapter.supports_api_key is False
    with pytest.raises(BrasskeyError, match="command line"):
        SglangAdapter().build_launch(
            install, local_model(tmp_path), options(api_key_env="BRASSKEY_ENGINE_KEY"), L4
        )


def test_model_mount_and_gpus_option(tmp_path):
    snapshot = tmp_path / "models--a--b" / "snapshots" / "rev1"
    mount = model_mount(snapshot)
    assert mount.host_dir == tmp_path / "models--a--b"
    assert mount.model_path == "/models/models--a--b/snapshots/rev1"
    plain = model_mount(tmp_path / "my-model")
    assert plain.model_path == "/models/my-model"
    assert gpus_option((0, 1)) == '"device=0,1"'
    assert gpus_option(()) == "all"


def test_paths_with_colons_cannot_be_mounted(tmp_path):
    bad = tmp_path / "a:b" / "models--x--y" / "snapshots" / "r"
    bad.mkdir(parents=True)
    model = local_model(tmp_path).model_copy(update={"snapshot_dir": bad})
    with pytest.raises(BrasskeyError, match="':'"):
        VllmAdapter().build_launch(docker_install(EngineId.VLLM, VLLM_IMAGE), model, options(), L4)


# --- failures ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0",
         "driver is too old"),
        ("docker: Error response from daemon: could not select device driver \"\" with "
         "capabilities: [[gpu]]", "Docker can't use the NVIDIA GPUs"),
        ("permission denied while trying to connect to the Docker daemon socket",
         "permission denied"),
        ("Unable to find image 'vllm/vllm-openai@sha256:...' locally", "isn't on this machine"),
        ("ValueError: Free memory on device (10.1/22.0 GiB) on startup is less than desired GPU "
         "memory utilization (0.9, 19.8 GiB)", "Other programs are using GPU memory"),
        ("ValueError: To serve at least one request with the models's max seq len (40960), "
         "(4.38 GiB KV cache is needed), which is larger than the available KV cache memory. "
         "Based on the available memory, the estimated maximum model length is 9000. Try "
         "increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing "
         "the engine.", "too little GPU memory left"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
         "ran out of GPU memory"),
        ("RuntimeError: NCCL error: unhandled system error", "NCCL"),
        ("ValueError: Model architectures ['FooForCausalLM'] are not supported for now.",
         "doesn't support this model"),
    ],
)  # fmt: skip
def test_failures_are_explained(line, expected):
    for cls in (VllmAdapter, SglangAdapter):
        error = cls().explain_failure(1, ["INFO starting", line])
        assert error is not None, line
        assert expected in error.message
        assert error.fix


def test_driver_fix_names_the_driver_version():
    error = VllmAdapter().explain_failure(1, ["unsatisfied condition: cuda>=13.0"])
    assert error is not None
    assert "580" in (error.fix or "")


def test_unknown_failures_show_the_log_instead():
    assert VllmAdapter().explain_failure(1, ["something new went wrong"]) is None


# --- NVIDIA rules --------------------------------------------------------------------------------


def test_choose_gpus_prefers_free_memory():
    hw = machines.l4(count=4)
    busy = tuple(
        gpu.model_copy(update={"memory_free_bytes": 1}) if gpu.index in (0, 1) else gpu
        for gpu in hw.accelerators
    )
    assert choose_gpus(busy, 2) == (2, 3)
    assert choose_gpus(hw.accelerators, 1) == (0,)


def test_old_driver_is_a_warning_with_the_driver_to_install():
    old = with_driver(machines.l4(), "12.8", "570.172.08")
    check = check_cuda_driver(EngineId.VLLM, "13.0", old)
    assert check.supported
    (note,) = check.notes
    assert "570.172.08" in note
    assert "driver 580 or newer" in note
    assert check_cuda_driver(EngineId.VLLM, "13.0", L4).notes == ()
    assert check_cuda_driver(EngineId.LLAMACPP, "13.0", old).notes == ()


def test_mig_gpus_are_not_planned_for_vllm():
    hw = machines.h100()
    mig = hw.model_copy(
        update={"accelerators": (hw.accelerators[0].model_copy(update={"mig_enabled": True}),)}
    )
    check = check_engine(EngineId.VLLM, mig)
    assert not check.supported
    assert "MIG" in check.reasons[0]


def test_volta_is_too_old_for_cuda_13_builds():
    v100 = machines.nvidia("Tesla V100-SXM2-16GB", 16384, (7, 0), 900)
    assert not check_engine(EngineId.VLLM, v100).supported
    assert check_engine(EngineId.VLLM, machines.t4()).supported


def test_parse_cuda_version():
    assert parse_cuda_version("13.0") == (13, 0)
    assert parse_cuda_version("12.9.1") == (12, 9)
    assert parse_cuda_version("x") is None
    assert parse_cuda_version(None) is None


def test_every_gpu_in_the_fake_machines_is_nvidia():
    assert all(acc.kind == AcceleratorKind.NVIDIA_GPU for acc in L4.accelerators)
