# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon GPU probe (read-only: ``sysctl`` and ``ioreg``).

On Apple Silicon the GPU shares unified memory with the CPU, but macOS only lets the GPU wire part
of it. That limit is what decides whether a model fits, so it is reported as the accelerator's
memory:

- if ``iogpu.wired_limit_mb`` is set (non-zero), that exact value;
- otherwise macOS's default, which Metal reports as ``recommendedMaxWorkingSetSize``: about 2/3 of
  RAM on Macs with less than 36 GiB and about 3/4 from 36 GiB up. This is labeled an estimate
  because reading the exact value needs the Metal API, which Brasskey doesn't load.
"""

import logging
import re
import sys

from brasskey.hardware.commands import IOREG, CommandRunner, parse_int, read_sysctl, run_command
from brasskey.hardware.types import (
    GIB,
    MIB,
    Accelerator,
    AcceleratorKind,
    ProbeIssue,
    ProbeResult,
)

logger = logging.getLogger(__name__)

PROBE_NAME = "apple"

APPLE_KEYS = ("machdep.cpu.brand_string", "hw.memsize", "hw.optional.arm64", "iogpu.wired_limit_mb")

# Peak unified-memory bandwidth in GB/s from Apple's published specifications. Chips sold with
# different bandwidth per GPU configuration are keyed by GPU core count.
APPLE_BANDWIDTH_GBS: dict[str, float | dict[int, float]] = {
    "M1": 68.25,
    "M1 Pro": 200,
    "M1 Max": 400,
    "M1 Ultra": 800,
    "M2": 100,
    "M2 Pro": 200,
    "M2 Max": 400,
    "M2 Ultra": 800,
    "M3": 100,
    "M3 Pro": 150,
    "M3 Max": {30: 300, 40: 400},
    "M3 Ultra": 819,
    "M4": 120,
    "M4 Pro": 273,
    "M4 Max": {32: 410, 40: 546},
    "M5": 153,
}

LARGE_MAC_BYTES = 36 * GIB
_GPU_CORES = re.compile(r'"gpu-core-count"\s*=\s*(\d+)')


def apple_bandwidth_gbs(chip: str, gpu_cores: int | None) -> float | None:
    """Published bandwidth for a chip such as "M3 Pro"; None when unknown or ambiguous."""
    entry = APPLE_BANDWIDTH_GBS.get(chip)
    if isinstance(entry, dict):
        return entry.get(gpu_cores) if gpu_cores is not None else None
    return entry


def chip_from_brand(brand: str) -> tuple[str, bool]:
    """("M3 Pro", virtual?) from a brand string such as "Apple M3 Pro" or "Apple M1 (Virtual)"."""
    virtual = "(virtual)" in brand.lower()
    chip = brand.replace("(Virtual)", "").replace("Apple", "", 1).strip()
    return chip, virtual


def default_gpu_memory_limit(memsize: int) -> int:
    """macOS's default GPU working-set limit (estimate; see module docstring)."""
    fraction = 3 / 4 if memsize >= LARGE_MAC_BYTES else 2 / 3
    return int(memsize * fraction)


class AppleSiliconProbe:
    """Reports the Apple GPU on Apple Silicon Macs."""

    name = PROBE_NAME

    def __init__(self, *, platform_name: str | None = None, run: CommandRunner = run_command):
        self._platform = platform_name or sys.platform
        self._run = run

    def is_applicable(self) -> bool:
        return self._platform == "darwin"

    def probe(self) -> ProbeResult:
        values = read_sysctl(self._run, APPLE_KEYS)
        if values.get("hw.optional.arm64") != "1":
            logger.debug("Intel Mac: no Apple GPU")
            return ProbeResult(
                issues=(
                    ProbeIssue(
                        probe=PROBE_NAME,
                        severity="info",
                        message="This Mac has an Intel CPU, so there is no Apple GPU to use; "
                        "Brasskey runs models on the CPU.",
                    ),
                )
            )

        memsize = parse_int(values.get("hw.memsize"))
        if not memsize:
            return ProbeResult(
                issues=(
                    ProbeIssue(
                        probe=PROBE_NAME,
                        severity="warning",
                        message="Couldn't read this Mac's memory size, so the Apple GPU is not "
                        "reported.",
                        fix="Check that `/usr/sbin/sysctl hw.memsize` works, then re-run.",
                    ),
                )
            )

        brand = values.get("machdep.cpu.brand_string") or "Apple Silicon"
        chip, virtual = chip_from_brand(brand)
        ioreg = self._run([IOREG, "-rd1", "-c", "AGXAccelerator"])
        match = _GPU_CORES.search(ioreg.stdout)
        gpu_cores = int(match.group(1)) if match else None

        notes: list[str] = []
        wired_mb = parse_int(values.get("iogpu.wired_limit_mb")) or 0
        if wired_mb > 0:
            limit = min(wired_mb * MIB, memsize)
            is_estimate = False
            notes.append("GPU memory limit set with iogpu.wired_limit_mb.")
        else:
            limit = default_gpu_memory_limit(memsize)
            is_estimate = True
            share = "3/4" if memsize >= LARGE_MAC_BYTES else "2/3"
            notes.append(
                f"macOS lets the GPU use about {share} of unified memory by default (estimate)."
            )

        # In a VM the paravirtual GPU's bandwidth says nothing about the host chip.
        bandwidth = None if virtual else apple_bandwidth_gbs(chip, gpu_cores)
        if bandwidth is not None:
            notes.append("Memory bandwidth from Apple's published specifications.")

        accelerator = Accelerator(
            kind=AcceleratorKind.APPLE_GPU,
            index=0,
            name=f"{brand} GPU",
            memory_total_bytes=limit,
            memory_is_estimate=is_estimate,
            unified_memory=True,
            gpu_cores=gpu_cores,
            mem_bandwidth_gbs=bandwidth,
            notes=tuple(notes),
        )
        logger.debug(
            "Apple GPU: %s, %s GPU cores, limit %.1f GiB (estimate=%s), bandwidth %s GB/s",
            chip,
            gpu_cores,
            limit / GIB,
            is_estimate,
            bandwidth,
        )
        return ProbeResult(accelerators=(accelerator,))
