# SPDX-License-Identifier: Apache-2.0
"""``brasskey doctor`` with fake hardware: human view, --json and issues."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from brasskey.engines.registry import runnable_engines
from brasskey.hardware import probe_hardware
from brasskey.hardware.apple import AppleSiliconProbe
from brasskey.hardware.cpu import CpuProbe
from brasskey.hardware.nvidia import NvmlProbe
from brasskey.hardware.types import GIB, OsInfo
from brasskey.runtime.state import ServerRecord
from fakes.commands import FakeRunner, load_sysctl, load_text, write_tree
from fakes.nvml import H100_SXM, FakeNvml, fleet
from fakes.system import FakeStats

MAC = OsInfo(family="darwin", version="15.4", description="macOS 15.4")
LINUX = OsInfo(family="linux", version="6.8.0", description="Ubuntu 24.04.1 LTS (kernel 6.8.0)")


def m3_pro_report():
    runner = FakeRunner(load_sysctl("macos-m3-pro.sysctl"), load_text("macos-m3-pro.ioreg"))
    stats = FakeStats(physical=11, logical=11, total=18 * GIB, available=6 * GIB)
    probes = [
        CpuProbe(platform_name="darwin", machine="arm64", run=runner, stats=stats),
        AppleSiliconProbe(platform_name="darwin", run=runner),
    ]
    return probe_hardware(probes, os_info=MAC)


def busy_h100_report(tmp_path):
    root = write_tree(tmp_path, {"proc/cpuinfo": load_text("linux-epyc-7r32.cpuinfo")})
    gpus = fleet(H100_SXM, 2)
    gpus[1] = replace(gpus[1], used_mib=40 * 1024)
    probes = [
        CpuProbe(
            platform_name="linux",
            machine="x86_64",
            stats=FakeStats(),
            proc_root=root / "proc",
            sys_root=root / "sys",
        ),
        NvmlProbe(FakeNvml(gpus), platform_name="linux", env={}, pci_check=lambda _r: True),
    ]
    return probe_hardware(probes, os_info=LINUX)


@pytest.fixture
def use_report(monkeypatch):
    """Fake hardware and no engines installed, whatever is on the machine running the tests."""
    monkeypatch.setattr(
        "brasskey.cli.doctor.detect_engines", lambda: dict.fromkeys(runnable_engines())
    )

    def install(report):
        monkeypatch.setattr("brasskey.cli.doctor.probe_hardware", lambda: report)
        return report

    return install


def test_doctor_shows_the_mac(cli, use_report):
    use_report(m3_pro_report())
    result = cli("doctor")
    assert result.exit_code == 0
    out = result.stdout
    assert "macOS 15.4" in out
    assert "Apple M3 Pro, 11 cores (5 performance, 6 efficiency), arm64" in out
    assert "features: bf16, dotprod, fp16, i8mm, neon" in out
    assert "18.0 GiB total, 6.0 GiB available" in out
    assert "Apple M3 Pro GPU, 14 GPU cores" in out
    assert "about 12.0 GiB of unified memory usable by the GPU (estimate)" in out
    assert "150 GB/s peak bandwidth" in out
    assert "Fingerprint: sha256:" in out
    assert "llama.cpp (not installed: brew install llama.cpp)" in out
    assert "vLLM: not for this machine. vLLM runs on Linux (or WSL2) with an NVIDIA GPU." in out
    # The engine `brasskey run` would pick here first: MLX, which uv sets up.
    assert result.last_line() == "Next: brew install uv (install an engine)"


def test_doctor_shows_gpu_issues_with_fixes(cli, use_report, tmp_path):
    use_report(busy_h100_report(tmp_path))
    result = cli("doctor")
    assert result.exit_code == 0  # issues inform; they don't fail the command
    assert "GPU 1" in result.stdout
    assert "compute 9.0" in result.stdout
    assert "Things to know" in result.stdout
    assert "in use by other processes" in result.stdout
    assert "Fix: Stop the other GPU processes" in result.stdout


def test_doctor_lists_running_servers_by_engine_name(cli, use_report, monkeypatch):
    use_report(m3_pro_report())
    record = ServerRecord(
        id="srv", pid=4242, process_started_at=1.0, engine="llamacpp", engine_version="1",
        model_id="qwen3-0.6b", host="127.0.0.1", port=8000, engine_port=40000, tls=False,
        base_url="http://127.0.0.1:8000/v1", auth="none",
        started_at=datetime(2026, 11, 8, 15, 13, tzinfo=UTC), log_path="server.log",
        brasskey_version="0",
    )  # fmt: skip
    monkeypatch.setattr("brasskey.cli.doctor.list_records", lambda _paths: [record])
    out = cli("doctor").stdout
    assert "qwen3-0.6b at http://127.0.0.1:8000/v1 (llama.cpp, pid 4242, since 2026-11-08" in out


def test_doctor_json(cli, use_report):
    report = use_report(m3_pro_report())
    document = cli("doctor", "--json").json()
    assert document["ok"] is True
    assert document["data"]["fingerprint"] == report.fingerprint()
    hardware = document["data"]["hardware"]
    assert hardware["cpu"]["model_name"] == "Apple M3 Pro"
    assert hardware["accelerators"][0]["memory_is_estimate"] is True


def test_doctor_without_gpus_says_cpu(cli, use_report):
    use_report(probe_hardware([], os_info=LINUX))
    assert "none found; models will run on the CPU" in cli("doctor").stdout
