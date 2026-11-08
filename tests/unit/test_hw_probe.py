# SPDX-License-Identifier: Apache-2.0
"""Merging probe results, crash isolation, fingerprints and OS detection."""

import json
from dataclasses import replace

from brasskey.hardware import probe_hardware
from brasskey.hardware.cpu import CpuProbe
from brasskey.hardware.nvidia import NvmlProbe
from brasskey.hardware.osinfo import detect_os
from brasskey.hardware.types import GIB, OsInfo, ProbeResult
from fakes.commands import load_text, write_tree
from fakes.nvml import H100_SXM, L4, FakeNvml, fleet
from fakes.system import FakeStats

LINUX = OsInfo(family="linux", version="6.8.0", description="Ubuntu 24.04.1 LTS (kernel 6.8.0)")


class StaticProbe:
    def __init__(self, name, result=None, *, applicable=True, crash=False):
        self.name = name
        self._result = result or ProbeResult()
        self._applicable = applicable
        self._crash = crash
        self.called = False

    def is_applicable(self):
        return self._applicable

    def probe(self):
        self.called = True
        if self._crash:
            raise RuntimeError("bug")
        return self._result


def linux_gpu_box(tmp_path, gpus, *, stats=None, env=None):
    root = write_tree(tmp_path, {"proc/cpuinfo": load_text("linux-epyc-7r32.cpuinfo")})
    probes = [
        CpuProbe(
            platform_name="linux",
            machine="x86_64",
            stats=stats or FakeStats(physical=48, logical=96, total=1024 * GIB),
            proc_root=root / "proc",
            sys_root=root / "sys",
        ),
        NvmlProbe(FakeNvml(gpus), platform_name="linux", env=env or {}, pci_check=lambda _r: True),
    ]
    return probe_hardware(probes, os_info=LINUX)


def test_merges_cpu_memory_and_gpus(tmp_path):
    report = linux_gpu_box(tmp_path, fleet(H100_SXM, 8))
    assert report.cpu.model_name == "AMD EPYC 7R32"
    assert report.memory.total_bytes == 1024 * GIB
    assert len(report.accelerators) == 8
    assert report.os == LINUX


def test_crashing_probe_becomes_a_warning_and_others_still_run():
    crashing = StaticProbe("broken", crash=True)
    later = StaticProbe("later")
    report = probe_hardware([crashing, later], os_info=LINUX)
    assert later.called
    (issue, *_rest) = report.issues
    assert issue.probe == "broken"
    assert issue.severity == "warning"
    assert "RuntimeError" in issue.message


def test_skips_probes_that_do_not_apply():
    skipped = StaticProbe("nope", applicable=False)
    probe_hardware([skipped], os_info=LINUX)
    assert skipped.called is False


def test_fallback_when_no_probe_reports_cpu_or_memory():
    report = probe_hardware([], os_info=LINUX)
    assert report.cpu.logical_cores >= 1
    assert report.memory.total_bytes == 0
    assert any("memory" in issue.message for issue in report.issues)


def test_report_is_json_serializable_and_deterministic(tmp_path):
    report = linux_gpu_box(tmp_path, fleet(H100_SXM, 2))
    document = json.loads(report.model_dump_json())
    assert document["schema_version"] == 1
    assert document["accelerators"][0]["kind"] == "nvidia_gpu"
    assert document["cpu"]["features"] == sorted(document["cpu"]["features"])


def test_fingerprint_is_stable_and_ignores_volatile_fields(tmp_path):
    first = linux_gpu_box(tmp_path / "a", fleet(H100_SXM, 2))
    busy = [replace(gpu, used_mib=40 * 1024) for gpu in fleet(H100_SXM, 2)]
    second = linux_gpu_box(
        tmp_path / "b",
        busy,
        stats=FakeStats(physical=48, logical=96, total=1024 * GIB, available=1),
    )
    assert second.issues  # the busy GPUs produced warnings...
    assert first.fingerprint() == second.fingerprint()  # ...but the hardware is the same
    assert first.fingerprint().startswith("sha256:")


def test_fingerprint_changes_with_the_hardware(tmp_path):
    h100s = linux_gpu_box(tmp_path / "a", fleet(H100_SXM, 2))
    l4s = linux_gpu_box(tmp_path / "b", fleet(L4, 2))
    one_h100 = linux_gpu_box(tmp_path / "c", fleet(H100_SXM, 1))
    assert len({h100s.fingerprint(), l4s.fingerprint(), one_h100.fingerprint()}) == 3


def test_detect_os_linux_with_os_release(tmp_path):
    etc = write_tree(tmp_path, {"os-release": load_text("ubuntu-24.04.os-release")})
    info = detect_os(platform_name="linux", release="6.8.0-1012-aws", etc_root=etc)
    assert info.family == "linux"
    assert info.version == "6.8.0-1012-aws"
    assert info.description == "Ubuntu 24.04.1 LTS (kernel 6.8.0-1012-aws)"


def test_detect_os_wsl_and_missing_os_release(tmp_path):
    info = detect_os(
        platform_name="linux", release="5.15.153.1-microsoft-standard-WSL2", etc_root=tmp_path
    )
    assert info.description.startswith("Linux (kernel")
    assert info.description.endswith("on WSL2")


def test_detect_os_macos_and_others():
    assert detect_os(platform_name="darwin", mac_version="15.4").description == "macOS 15.4"
    assert detect_os(platform_name="win32", release="11").family == "windows"
    assert detect_os(platform_name="freebsd14", release="14.1").family == "other"
