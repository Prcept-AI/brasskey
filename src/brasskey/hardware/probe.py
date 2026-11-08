# SPDX-License-Identifier: Apache-2.0
"""Run the hardware probes and merge their results into one :class:`HardwareReport`.

Probing never fails as a whole: a probe that crashes (a bug) becomes a warning in the report and
the other probes still run, so ``doctor`` and ``run`` always get an answer.
"""

import logging
import os
import platform
import time
from collections.abc import Sequence
from typing import Protocol

from brasskey.errors import ISSUES_URL
from brasskey.hardware.apple import AppleSiliconProbe
from brasskey.hardware.cpu import CpuProbe, normalize_arch
from brasskey.hardware.nvidia import NvmlProbe
from brasskey.hardware.osinfo import detect_os
from brasskey.hardware.types import (
    Accelerator,
    CpuInfo,
    HardwareReport,
    MemoryInfo,
    OsInfo,
    ProbeIssue,
    ProbeResult,
)

logger = logging.getLogger(__name__)


class HardwareProbe(Protocol):
    """A read-only probe for one kind of hardware."""

    @property
    def name(self) -> str: ...

    def is_applicable(self) -> bool:
        """Cheap check (platform, presence) before probing."""
        ...

    def probe(self) -> ProbeResult:
        """Read the hardware. Expected failures are returned as issues, never raised."""
        ...


def default_probes() -> list[HardwareProbe]:
    """The probes for this machine: CPU and memory first, then accelerators."""
    return [CpuProbe(), AppleSiliconProbe(), NvmlProbe()]


def _fallback_cpu() -> CpuInfo:
    count = os.cpu_count() or 1
    return CpuInfo(
        arch=normalize_arch(platform.machine()),
        model_name=platform.processor() or "Unknown CPU",
        physical_cores=count,
        logical_cores=count,
    )


def probe_hardware(
    probes: Sequence[HardwareProbe] | None = None, *, os_info: OsInfo | None = None
) -> HardwareReport:
    """Probe this machine (read-only) and return a merged report."""
    selected = list(probes) if probes is not None else default_probes()
    accelerators: list[Accelerator] = []
    issues: list[ProbeIssue] = []
    cpu: CpuInfo | None = None
    memory: MemoryInfo | None = None

    for probe in selected:
        if not probe.is_applicable():
            logger.debug("Probe %s not applicable here", probe.name)
            continue
        started = time.perf_counter()
        try:
            result = probe.probe()
        except Exception as error:  # a bug in one probe must not break the report
            logger.debug("Probe %s crashed", probe.name, exc_info=True)
            issues.append(
                ProbeIssue(
                    probe=probe.name,
                    severity="warning",
                    message=f"The {probe.name} probe failed unexpectedly "
                    f"({type(error).__name__}); its hardware is not reported.",
                    fix=f"Re-run with --debug and report it at {ISSUES_URL}.",
                )
            )
            continue
        logger.debug(
            "Probe %s: %d accelerator(s), %d issue(s) in %.0f ms",
            probe.name,
            len(result.accelerators),
            len(result.issues),
            (time.perf_counter() - started) * 1000,
        )
        accelerators.extend(result.accelerators)
        issues.extend(result.issues)
        cpu = cpu or result.cpu
        memory = memory or result.memory

    if cpu is None:
        cpu = _fallback_cpu()
    if memory is None:
        memory = MemoryInfo(total_bytes=0, available_bytes=0)
        issues.append(
            ProbeIssue(
                probe="cpu",
                severity="warning",
                message="Couldn't read the system memory size.",
                fix=f"Re-run with --debug and report it at {ISSUES_URL}.",
            )
        )

    return HardwareReport(
        os=os_info or detect_os(),
        cpu=cpu,
        memory=memory,
        accelerators=tuple(accelerators),
        issues=tuple(issues),
    )
