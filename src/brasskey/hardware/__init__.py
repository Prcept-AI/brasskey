# SPDX-License-Identifier: Apache-2.0
"""Read-only hardware probes: NVIDIA GPUs (NVML), Apple Silicon, and x86/ARM CPUs.

Use :func:`probe_hardware` to get a :class:`HardwareReport` for this machine.
"""

from brasskey.hardware.probe import HardwareProbe, default_probes, probe_hardware
from brasskey.hardware.types import (
    Accelerator,
    AcceleratorKind,
    CpuInfo,
    HardwareReport,
    MemoryInfo,
    OsInfo,
    ProbeIssue,
    ProbeResult,
)

__all__ = [
    "Accelerator",
    "AcceleratorKind",
    "CpuInfo",
    "HardwareProbe",
    "HardwareReport",
    "MemoryInfo",
    "OsInfo",
    "ProbeIssue",
    "ProbeResult",
    "default_probes",
    "probe_hardware",
]
