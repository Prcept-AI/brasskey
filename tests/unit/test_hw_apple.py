# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon GPU probe, replaying sysctl/ioreg captured from real Macs."""

import pytest

from brasskey.hardware.apple import (
    AppleSiliconProbe,
    apple_bandwidth_gbs,
    chip_from_brand,
    default_gpu_memory_limit,
)
from brasskey.hardware.types import GIB, MIB, AcceleratorKind
from fakes.commands import FakeRunner, load_sysctl, load_text

M3_PRO = load_sysctl("macos-m3-pro.sysctl")
M3_PRO_IOREG = load_text("macos-m3-pro.ioreg")


def _probe(sysctl, ioreg=""):
    return AppleSiliconProbe(platform_name="darwin", run=FakeRunner(sysctl, ioreg)).probe()


def test_m3_pro_captured_from_a_real_mac():
    result = _probe(M3_PRO, M3_PRO_IOREG)
    assert result.issues == ()
    (gpu,) = result.accelerators
    assert gpu.kind == AcceleratorKind.APPLE_GPU
    assert gpu.name == "Apple M3 Pro GPU"
    assert gpu.gpu_cores == 14
    assert gpu.unified_memory is True
    assert gpu.memory_free_bytes is None
    # 18 GiB Mac, no iogpu.wired_limit_mb: macOS default of about 2/3, labeled an estimate.
    assert gpu.memory_total_bytes == 12 * GIB
    assert gpu.memory_is_estimate is True
    assert gpu.mem_bandwidth_gbs == 150
    assert any("estimate" in note for note in gpu.notes)


def test_wired_limit_set_by_the_user_is_exact():
    sysctl = {**M3_PRO, "hw.memsize": str(64 * GIB), "iogpu.wired_limit_mb": "57344"}
    (gpu,) = _probe(sysctl, M3_PRO_IOREG).accelerators
    assert gpu.memory_total_bytes == 57344 * MIB
    assert gpu.memory_is_estimate is False


def test_wired_limit_never_exceeds_ram():
    sysctl = {**M3_PRO, "iogpu.wired_limit_mb": str(10**7)}
    (gpu,) = _probe(sysctl, M3_PRO_IOREG).accelerators
    assert gpu.memory_total_bytes == int(M3_PRO["hw.memsize"])


@pytest.mark.parametrize(
    ("memsize_gib", "expected_gib"),
    [(8, 8 * 2 / 3), (16, 16 * 2 / 3), (32, 32 * 2 / 3), (36, 27), (64, 48), (128, 96)],
)
def test_default_gpu_memory_limit(memsize_gib, expected_gib):
    assert default_gpu_memory_limit(memsize_gib * GIB) == int(expected_gib * GIB)


def test_m4_max_bandwidth_depends_on_gpu_cores():
    sysctl = {**M3_PRO, "machdep.cpu.brand_string": "Apple M4 Max", "hw.memsize": str(128 * GIB)}
    (gpu,) = _probe(sysctl, '"gpu-core-count" = 40').accelerators
    assert gpu.mem_bandwidth_gbs == 546
    (binned,) = _probe(sysctl, '"gpu-core-count" = 32').accelerators
    assert binned.mem_bandwidth_gbs == 410


def test_unknown_gpu_core_count_leaves_ambiguous_bandwidth_unset():
    sysctl = {**M3_PRO, "machdep.cpu.brand_string": "Apple M3 Max"}
    (gpu,) = _probe(sysctl, "").accelerators
    assert gpu.gpu_cores is None
    assert gpu.mem_bandwidth_gbs is None


def test_unknown_future_chip_has_no_bandwidth():
    sysctl = {**M3_PRO, "machdep.cpu.brand_string": "Apple M9 Hyper"}
    (gpu,) = _probe(sysctl, M3_PRO_IOREG).accelerators
    assert gpu.name == "Apple M9 Hyper GPU"
    assert gpu.mem_bandwidth_gbs is None


def test_virtual_mac_reports_gpu_without_host_bandwidth():
    (gpu,) = _probe(load_sysctl("macos-m1-virtual.sysctl")).accelerators
    assert gpu.name == "Apple M1 (Virtual) GPU"
    assert gpu.mem_bandwidth_gbs is None
    assert gpu.gpu_cores is None


def test_intel_mac_has_no_apple_gpu():
    result = _probe(load_sysctl("macos-intel-i9.sysctl"))
    assert result.accelerators == ()
    (issue,) = result.issues
    assert issue.severity == "info"
    assert "Intel CPU" in issue.message


def test_missing_memsize_is_a_warning_not_a_crash():
    sysctl = {key: value for key, value in M3_PRO.items() if key != "hw.memsize"}
    result = _probe(sysctl, M3_PRO_IOREG)
    assert result.accelerators == ()
    assert result.issues[0].severity == "warning"


@pytest.mark.parametrize("platform_name", ["linux", "win32"])
def test_only_applies_on_macos(platform_name):
    assert not AppleSiliconProbe(platform_name=platform_name).is_applicable()


@pytest.mark.parametrize(
    ("brand", "chip", "virtual"),
    [
        ("Apple M3 Pro", "M3 Pro", False),
        ("Apple M1 (Virtual)", "M1", True),
        ("Apple M4", "M4", False),
    ],
)
def test_chip_from_brand(brand, chip, virtual):
    assert chip_from_brand(brand) == (chip, virtual)


def test_bandwidth_table_lookup():
    assert apple_bandwidth_gbs("M1", None) == 68.25
    assert apple_bandwidth_gbs("M3 Max", 30) == 300
    assert apple_bandwidth_gbs("M3 Max", 36) is None
    assert apple_bandwidth_gbs("Unknown", 10) is None
