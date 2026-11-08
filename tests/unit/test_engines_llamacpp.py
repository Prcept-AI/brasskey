# SPDX-License-Identifier: Apache-2.0
"""llama.cpp adapter: detection from captured real output, launch goldens, failures, hints."""

from pathlib import Path

import pytest

from brasskey.engines import llamacpp
from brasskey.engines.base import EngineInstall, LocalModel, ServeOptions
from brasskey.engines.llamacpp import LlamaCppAdapter, parse_devices
from brasskey.engines.registry import (
    METAL_NOTE,
    detect_engines,
    refine_hardware,
    reported_gpu_backends,
)
from brasskey.engines.types import EngineId
from brasskey.hardware.commands import CommandOutput
from brasskey.hardware.types import GIB, MIB
from fakes import machines
from fakes.commands import load_text
from fakes.engine import TINY_ARTIFACT

VERSION = load_text("../engines/llamacpp-0.9.0.version")
HELP = load_text("../engines/llamacpp-0.9.0.help")
DEVICES = load_text("../engines/llamacpp-0.9.0-m3pro.devices")


def runner(*, version=VERSION, help_text=HELP, devices=DEVICES):
    outputs = {"--version": version, "--help": help_text, "--list-devices": devices}

    def run(argv, *, timeout_s=5.0):
        return CommandOutput(0, "", outputs[argv[1]])  # llama-server prints to stderr

    return run


def detect(**kwargs) -> EngineInstall | None:
    return LlamaCppAdapter(run=runner(**kwargs), which="/opt/homebrew/bin/llama-server").detect()


def installed() -> EngineInstall:
    install = detect()
    assert install is not None
    return install


def test_detects_the_real_0_9_0_output():
    install = detect()
    assert install is not None
    assert install.version == "0.9.0 (build 12110, commit cd959bae9)"
    assert install.executable == ("/opt/homebrew/bin/llama-server",)
    assert {
        "--jinja",
        "--metrics",
        "--no-webui",
        "--offline",
        "--fit",
        "--api-key",
    } <= install.flags
    assert install.backends == ("metal",)
    (device,) = install.devices
    assert (device.name, device.memory_total_bytes) == ("Apple M3 Pro", 12288 * MIB)


def test_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp.shutil, "which", lambda _name: None)
    monkeypatch.setattr(llamacpp, "FALLBACK_DIRS", (tmp_path,))
    assert LlamaCppAdapter().detect() is None


def test_unrecognised_version_output_means_not_usable():
    assert detect(version="garbage") is None


def test_parse_cuda_and_vulkan_devices_and_skip_blas():
    text = (
        "Available devices:\n"
        "  BLAS: OpenBLAS (0 MiB, 0 MiB free)\n"
        "  CUDA0: NVIDIA H100 80GB HBM3 (81559 MiB, 80950 MiB free)\n"
        "  Vulkan0: AMD Radeon (16368 MiB, 16000 MiB free)\n"
    )
    devices = parse_devices(text)
    assert [(d.backend, d.name) for d in devices] == [
        ("cuda", "NVIDIA H100 80GB HBM3"),
        ("vulkan", "AMD Radeon"),
    ]


# --list-devices of a CUDA build and of CPU-only builds (llama.cpp b12110, common/arg.cpp:
# common_print_available_devices skips CPU devices and prints "(none)" when nothing is left).
CUDA_DEVICES = "Available devices:\n  CUDA0: NVIDIA H100 80GB HBM3 (81559 MiB, 80950 MiB free)\n"
CPU_ONLY_DEVICES = "Available devices:\n  (none)\n"
CPU_BLAS_DEVICES = "Available devices:\n  BLAS: OpenBLAS (0 MiB, 0 MiB free)\n"


def test_the_device_list_tells_gpu_builds_from_cpu_only_builds():
    assert installed().gpu_backends == frozenset({"metal"})
    cuda = detect(devices=CUDA_DEVICES)
    assert cuda is not None
    assert cuda.gpu_backends == frozenset({"cuda"})
    for listing in (CPU_ONLY_DEVICES, CPU_BLAS_DEVICES):
        cpu_only = detect(devices=listing)
        assert cpu_only is not None
        assert cpu_only.devices == ()
        assert cpu_only.devices_listed is True
        assert cpu_only.gpu_backends == frozenset()  # measured: no GPU backend at all


def test_backends_stay_unknown_without_a_complete_device_list():
    def listing_fails(argv, *, timeout_s=5.0):
        if argv[1] == "--list-devices":
            return CommandOutput(1, "", "ggml_cuda_init: failed to initialize CUDA\n")
        return runner()(argv, timeout_s=timeout_s)

    failed = LlamaCppAdapter(run=listing_fails, which="/usr/local/bin/llama-server").detect()
    assert failed is not None
    assert failed.gpu_backends is None  # not proof of a CPU-only build
    unfamiliar = detect(devices="something else entirely\n")
    assert unfamiliar is not None
    assert unfamiliar.gpu_backends is None
    old = detect(help_text=HELP.replace("--list-devices", "--list-things"))
    assert old is not None
    assert "--list-devices" not in old.flags
    assert old.gpu_backends is None


def local_model() -> LocalModel:
    gguf = Path("/cache/snapshots/x/tiny-Q4_K_M.gguf")
    return LocalModel(artifact=TINY_ARTIFACT, snapshot_dir=gguf.parent, files=(gguf,))


def options(**overrides):
    values = {"port": 8000, "served_model_name": "qwen3-4b", "context_length": 8192}
    return ServeOptions(**(values | overrides))


def test_launch_golden_on_apple_silicon():
    install = installed()
    spec = LlamaCppAdapter().build_launch(install, local_model(), options(), machines.m3_pro_18gb())
    assert spec.argv == (
        "/opt/homebrew/bin/llama-server",
        "--model", "/cache/snapshots/x/tiny-Q4_K_M.gguf",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--ctx-size", "8192",
        "--parallel", "1",
        "--alias", "qwen3-4b",
        "--n-gpu-layers", "999",
        "--jinja", "--metrics", "--no-webui", "--offline", "--fit", "off",
    )  # fmt: skip
    assert spec.base_url == "http://127.0.0.1:8000/v1"
    assert spec.health_url == "http://127.0.0.1:8000/health"
    assert spec.metrics_url == "http://127.0.0.1:8000/metrics"
    assert spec.env["DO_NOT_TRACK"] == "1"
    assert spec.env["HF_HUB_OFFLINE"] == "1"
    assert spec.secret_env == {}


def test_launch_on_cpu_keeps_layers_on_the_cpu_and_slots_share_context():
    install = installed()
    spec = LlamaCppAdapter().build_launch(
        install, local_model(), options(parallel=4), machines.cpu_only(32)
    )
    argv = list(spec.argv)
    assert argv[argv.index("--n-gpu-layers") + 1] == "0"
    assert argv[argv.index("--ctx-size") + 1] == str(8192 * 4)


@pytest.mark.parametrize(
    ("devices", "listed", "layers"),
    [
        (CUDA_DEVICES, True, "999"),
        (CPU_ONLY_DEVICES, True, "0"),  # a CPU-only build on a GPU machine
        (CPU_ONLY_DEVICES, False, "999"),  # unknown: assume the GPU, as before
    ],
)
def test_gpu_layers_follow_what_the_build_can_use_on_an_nvidia_machine(devices, listed, layers):
    install = detect(devices=devices)
    assert install is not None
    install = install.model_copy(update={"devices_listed": listed})
    spec = LlamaCppAdapter().build_launch(install, local_model(), options(), machines.h100())
    argv = list(spec.argv)
    assert argv[argv.index("--n-gpu-layers") + 1] == layers


def test_older_versions_get_only_the_flags_they_know():
    old = EngineInstall(
        engine=EngineId.LLAMACPP,
        version="b4000",
        executable=("llama-server",),
        flags=frozenset({"--jinja"}),
    )
    spec = LlamaCppAdapter().build_launch(old, local_model(), options(), machines.m3_pro_18gb())
    assert "--jinja" in spec.argv
    for flag in ("--no-webui", "--fit", "--offline", "--metrics"):
        assert flag not in spec.argv
    assert spec.metrics_url is None


def test_api_key_goes_through_the_environment_and_ipv6_urls_are_bracketed():
    spec = LlamaCppAdapter().build_launch(
        installed(),
        local_model(),
        options(host="::1", api_key_env="BRASSKEY_API_KEY"),
        machines.m3_pro_18gb(),
    )
    assert spec.secret_env == {"LLAMA_API_KEY": "BRASSKEY_API_KEY"}
    assert "--api-key" not in spec.argv
    assert spec.base_url == "http://[::1]:8000/v1"


@pytest.mark.parametrize(
    ("log_line", "message"),
    [
        ("ggml_metal: failed to allocate buffer of size 9000 MiB", "ran out of memory"),
        ("CUDA error: out of memory", "ran out of memory"),
        ("couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8000", "already in use"),
        (
            "llama_model_load: error loading model: unknown model architecture: 'qwen9'",
            "doesn't support this model",
        ),
        ("error: invalid argument: --fit", "rejected an option"),
        ("gguf_init_from_file: failed to read magic", "couldn't load the model file"),
    ],
)
def test_explain_failure(log_line, message):
    error = LlamaCppAdapter().explain_failure(1, ["loading...", log_line])
    assert error is not None
    assert message in error.message
    assert error.fix


def test_unknown_failure_is_left_to_the_log_tail():
    assert LlamaCppAdapter().explain_failure(1, ["something odd"]) is None


@pytest.mark.parametrize(
    ("hw", "command"),
    [
        (machines.m3_pro_18gb(), "brew install llama.cpp"),
        (machines.cpu_only(16), "brew install llama.cpp"),
    ],
)
def test_install_hints(hw, command):
    assert LlamaCppAdapter().install_hint(hw).commands[0] == command


def test_nvidia_hint_builds_the_tested_version_with_cuda_where_detect_finds_it():
    hint = LlamaCppAdapter().install_hint(machines.h100())
    (command,) = hint.commands  # one command: it becomes the `Next:` line
    assert command.startswith(
        "git clone --depth 1 --branch b12110 https://github.com/ggml-org/llama.cpp && "
    )
    # The tag is the version the fixtures captured (and the adapter was written against).
    assert f"(build {llamacpp.TESTED_BUILD.removeprefix('b')}, commit cd959bae9)" in VERSION
    assert "-DGGML_CUDA=ON" in command
    assert "-DBUILD_SHARED_LIBS=OFF" in command  # one self-contained binary to copy
    assert command.endswith("cp build/bin/llama-server ~/.local/bin/")
    assert Path.home() / ".local" / "bin" in llamacpp.FALLBACK_DIRS  # found even off PATH
    assert "docker" not in command  # a container image isn't something detect() can find
    assert hint.docs_url is not None
    assert hint.docs_url.endswith("build.md#cuda")


# --- registry -------------------------------------------------------------------------------------


def test_metal_measurement_replaces_the_macos_estimate():
    hw = machines.m3_pro_18gb()
    assert hw.accelerators[0].memory_is_estimate
    refined = refine_hardware(hw, {EngineId.LLAMACPP: detect()})
    gpu = refined.accelerators[0]
    assert gpu.memory_total_bytes == 12 * GIB
    assert gpu.memory_is_estimate is False
    assert METAL_NOTE in gpu.notes
    assert refine_hardware(hw, {EngineId.LLAMACPP: None}) == hw
    assert refine_hardware(machines.h100(), {EngineId.LLAMACPP: detect()}) == machines.h100()


def test_reported_gpu_backends_leave_out_engines_that_cannot_list_devices():
    cpu_only = detect(devices=CPU_ONLY_DEVICES)
    assert cpu_only is not None
    unknown = cpu_only.model_copy(update={"devices_listed": False})
    assert reported_gpu_backends({EngineId.LLAMACPP: cpu_only, EngineId.MLX: None}) == {
        EngineId.LLAMACPP: frozenset()
    }
    assert reported_gpu_backends({EngineId.LLAMACPP: installed()}) == {
        EngineId.LLAMACPP: frozenset({"metal"})
    }
    assert reported_gpu_backends({EngineId.LLAMACPP: unknown}) == {}


def test_detect_engines_with_injected_adapters():
    class Missing(LlamaCppAdapter):
        def detect(self):
            return None

    assert detect_engines({EngineId.LLAMACPP: Missing()}) == {EngineId.LLAMACPP: None}
