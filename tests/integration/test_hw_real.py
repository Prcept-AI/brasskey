# SPDX-License-Identifier: Apache-2.0
"""Probe the real machine running the tests (read-only).

The first test runs everywhere, including CI (macOS on Apple Silicon and Linux). The NVML test
needs a real NVIDIA GPU and runs only with ``pytest -m gpu``.
"""

import platform
import sys

import pytest

from brasskey.hardware import AcceleratorKind, probe_hardware
from brasskey.hardware.nvidia import NvmlProbe


def test_probe_this_machine():
    report = probe_hardware()
    assert report.cpu.logical_cores >= 1
    assert report.cpu.physical_cores >= 1
    assert report.memory.total_bytes > 0
    assert report.fingerprint().startswith("sha256:")
    if sys.platform == "darwin" and report.cpu.arch == "arm64":
        apple = report.accelerators_of(AcceleratorKind.APPLE_GPU)
        assert len(apple) == 1
        assert 0 < apple[0].memory_total_bytes <= report.memory.total_bytes
    if not report.cpu.rosetta:
        machine = platform.machine().lower()
        expected = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
        assert report.cpu.arch == expected


@pytest.mark.gpu
def test_real_nvidia_gpus():
    result = NvmlProbe().probe()
    assert result.accelerators, f"no NVIDIA GPU found: {result.issues}"
    for gpu in result.accelerators:
        assert gpu.memory_total_bytes > 0
        assert gpu.compute_capability is not None
        assert gpu.driver_version
