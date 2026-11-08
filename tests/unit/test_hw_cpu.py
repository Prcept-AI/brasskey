# SPDX-License-Identifier: Apache-2.0
"""CPU and memory probe: macOS (sysctl), Linux (/proc, cgroups), and other systems."""

import pytest

from brasskey.hardware.cpu import (
    CpuProbe,
    normalize_arch,
    parse_cpuinfo,
    read_cgroup_limits,
)
from brasskey.hardware.types import GIB
from fakes.commands import FakeRunner, load_sysctl, load_text, write_tree
from fakes.system import FakeStats

M3_PRO = load_sysctl("macos-m3-pro.sysctl")


def _mac(sysctl, *, machine="arm64", stats=None):
    probe = CpuProbe(
        platform_name="darwin", machine=machine, run=FakeRunner(sysctl), stats=stats or FakeStats()
    )
    return probe.probe()


def _linux(tmp_path, cpuinfo, *, machine="x86_64", stats=None, extra=None):
    files = {"proc/cpuinfo": load_text(cpuinfo), **(extra or {})}
    root = write_tree(tmp_path, files)
    probe = CpuProbe(
        platform_name="linux",
        machine=machine,
        stats=stats or FakeStats(),
        proc_root=root / "proc",
        sys_root=root / "sys",
    )
    return probe.probe()


# --- macOS ----------------------------------------------------------------------------------------


def test_m3_pro_captured_from_a_real_mac():
    result = _mac(M3_PRO, stats=FakeStats(total=18 * GIB, available=6 * GIB))
    cpu = result.cpu
    assert cpu is not None
    assert cpu.model_name == "Apple M3 Pro"
    assert cpu.arch == "arm64"
    assert (cpu.physical_cores, cpu.logical_cores) == (11, 11)
    assert (cpu.performance_cores, cpu.efficiency_cores) == (5, 6)
    assert cpu.features == ("bf16", "dotprod", "fp16", "i8mm", "neon")
    assert cpu.rosetta is False
    assert result.memory is not None
    assert result.memory.total_bytes == 18 * GIB
    assert result.issues == ()


def test_rosetta_reports_the_real_arch_and_warns():
    result = _mac({**M3_PRO, "sysctl.proc_translated": "1"}, machine="x86_64")
    assert result.cpu is not None
    assert result.cpu.arch == "arm64"  # the hardware, not the emulated Python
    assert result.cpu.rosetta is True
    (issue,) = result.issues
    assert issue.severity == "warning"
    assert "Rosetta" in issue.message
    assert issue.fix


def test_intel_mac_features_from_both_feature_leaves():
    result = _mac(load_sysctl("macos-intel-i9.sysctl"), machine="x86_64")
    cpu = result.cpu
    assert cpu is not None
    assert cpu.arch == "x86_64"
    assert cpu.features == ("avx", "avx2", "f16c", "fma")
    assert (cpu.physical_cores, cpu.logical_cores) == (8, 16)
    assert cpu.performance_cores is None


def test_sysctl_unavailable_falls_back_to_psutil_counts():
    result = _mac({}, machine="arm64", stats=FakeStats(physical=4, logical=8))
    assert result.cpu is not None
    assert (result.cpu.physical_cores, result.cpu.logical_cores) == (4, 8)


# --- Linux ----------------------------------------------------------------------------------------


def test_linux_amd_epyc_reads_only_the_first_processor_block(tmp_path):
    result = _linux(tmp_path, "linux-epyc-7r32.cpuinfo")
    cpu = result.cpu
    assert cpu is not None
    assert cpu.model_name == "AMD EPYC 7R32"
    assert cpu.features == ("avx", "avx2", "f16c", "fma")
    assert cpu.cpu_limit_cores is None
    assert result.issues == ()


def test_linux_sapphire_rapids_reports_avx512_and_amx(tmp_path):
    cpu = _linux(tmp_path, "linux-xeon-8488c.cpuinfo").cpu
    assert cpu is not None
    assert {"avx512f", "avx512_bf16", "avx512_vnni", "avx_vnni", "amx_tile", "amx_bf16"} <= set(
        cpu.features
    )


def test_linux_graviton3_names_the_core_and_maps_arm_features(tmp_path):
    cpu = _linux(tmp_path, "linux-graviton3.cpuinfo", machine="aarch64").cpu
    assert cpu is not None
    assert cpu.arch == "arm64"
    assert cpu.model_name == "ARM Neoverse-V1"
    assert cpu.features == ("bf16", "dotprod", "fp16", "i8mm", "neon", "sve")


def test_linux_unknown_arm_part_is_described_not_guessed(tmp_path):
    cpu = _linux(tmp_path, "linux-unknown-arm.cpuinfo", machine="aarch64").cpu
    assert cpu is not None
    assert cpu.model_name == "HiSilicon ARM CPU (part 0xd01)"


def test_linux_unknown_arm_implementer_keeps_raw_ids(tmp_path):
    cpuinfo = load_text("linux-unknown-arm.cpuinfo").replace("0x48", "0x99")
    root = write_tree(tmp_path, {"proc/cpuinfo": cpuinfo})
    probe = CpuProbe(
        platform_name="linux",
        machine="aarch64",
        stats=FakeStats(),
        proc_root=root / "proc",
        sys_root=root / "sys",
    )
    cpu = probe.probe().cpu
    assert cpu is not None
    assert cpu.model_name == "ARM CPU (implementer 0x99, part 0xd01)"


def test_linux_missing_cpuinfo_still_reports(tmp_path):
    probe = CpuProbe(
        platform_name="linux",
        machine="x86_64",
        stats=FakeStats(),
        proc_root=tmp_path / "nope",
        sys_root=tmp_path / "nope",
    )
    result = probe.probe()
    assert result.cpu is not None
    assert result.cpu.logical_cores == 16


CGROUP_V2_CONTAINER = {
    "proc/self/cgroup": "0::/\n",
    "sys/fs/cgroup/cgroup.controllers": "cpuset cpu io memory pids\n",
    "sys/fs/cgroup/memory.max": f"{8 * GIB}\n",
    "sys/fs/cgroup/cpu.max": "400000 100000\n",
}


def test_linux_container_limits_cgroup_v2(tmp_path):
    result = _linux(tmp_path, "linux-epyc-7r32.cpuinfo", extra=CGROUP_V2_CONTAINER)
    assert result.memory is not None
    assert result.memory.cgroup_limit_bytes == 8 * GIB
    assert result.memory.usable_total_bytes == 8 * GIB
    assert result.cpu is not None
    assert result.cpu.cpu_limit_cores == 4
    (issue,) = result.issues
    assert issue.severity == "info"
    assert "8.0 GiB" in issue.message
    assert "4 CPU cores" in issue.message


def test_cgroup_v2_nested_takes_the_tightest_limit(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "proc/self/cgroup": "0::/system.slice/app.service\n",
            "sys/fs/cgroup/cgroup.controllers": "memory cpu\n",
            "sys/fs/cgroup/memory.max": "max\n",
            "sys/fs/cgroup/system.slice/memory.max": f"{16 * GIB}\n",
            "sys/fs/cgroup/system.slice/app.service/memory.max": f"{12 * GIB}\n",
            "sys/fs/cgroup/system.slice/app.service/cpu.max": "max 100000\n",
        },
    )
    assert read_cgroup_limits(root / "proc", root / "sys") == (12 * GIB, None)


def test_cgroup_v1_limits_and_unlimited_marker(tmp_path):
    root = write_tree(
        tmp_path,
        {
            "proc/self/cgroup": "12:memory:/docker/abc\n",
            "sys/fs/cgroup/memory/memory.limit_in_bytes": f"{4 * GIB}\n",
            "sys/fs/cgroup/cpu/cpu.cfs_quota_us": "150000\n",
            "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
        },
    )
    assert read_cgroup_limits(root / "proc", root / "sys") == (4 * GIB, 1.5)

    unlimited = write_tree(
        tmp_path / "unlimited",
        {
            "sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
            "sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n",
            "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
        },
    )
    assert read_cgroup_limits(unlimited / "proc", unlimited / "sys") == (None, None)


def test_limit_above_host_ram_is_ignored(tmp_path):
    extra = {**CGROUP_V2_CONTAINER, "sys/fs/cgroup/memory.max": f"{512 * GIB}\n"}
    result = _linux(tmp_path, "linux-epyc-7r32.cpuinfo", extra=extra)
    assert result.memory is not None
    assert result.memory.cgroup_limit_bytes is None


def test_cpuset_affinity_limits_usable_cores(tmp_path):
    result = _linux(tmp_path, "linux-epyc-7r32.cpuinfo", stats=FakeStats(logical=16, usable=2))
    assert result.cpu is not None
    assert result.cpu.cpu_limit_cores == 2


# --- helpers and other systems ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("machine", "arch"),
    [
        ("x86_64", "x86_64"),
        ("AMD64", "x86_64"),
        ("arm64", "arm64"),
        ("aarch64", "arm64"),
        ("ppc64le", "other"),
    ],
)
def test_normalize_arch(machine, arch):
    assert normalize_arch(machine) == arch


def test_parse_cpuinfo_handles_empty_input():
    assert parse_cpuinfo(None) == {}
    assert parse_cpuinfo("\n\n") == {}


def test_generic_platform_uses_psutil_only():
    probe = CpuProbe(
        platform_name="win32", machine="AMD64", stats=FakeStats(physical=6, logical=12)
    )
    result = probe.probe()
    assert result.cpu is not None
    assert result.cpu.arch == "x86_64"
    assert result.cpu.logical_cores == 12
    assert result.memory is not None
