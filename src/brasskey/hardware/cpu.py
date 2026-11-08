# SPDX-License-Identifier: Apache-2.0
"""CPU and system memory probe.

- macOS: ``sysctl`` (true architecture even under Rosetta, performance/efficiency cores, ARM
  feature flags), psutil for memory.
- Linux: ``/proc/cpuinfo``, psutil, and container limits from cgroups (v1 and v2) and the CPU
  affinity mask, because psutil reports the host's RAM and cores inside a container.
- Elsewhere: psutil and ``platform`` only.

Only inference-relevant CPU features are reported (vector and matrix extensions that engines such
as llama.cpp use), which keeps reports short and fingerprints stable.
"""

import logging
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol

import psutil

from brasskey.hardware.commands import CommandRunner, parse_int, read_sysctl, run_command
from brasskey.hardware.types import GIB, CpuArch, CpuInfo, MemoryInfo, ProbeIssue, ProbeResult

logger = logging.getLogger(__name__)

PROBE_NAME = "cpu"

# x86 flags (as named in /proc/cpuinfo) that matter for CPU inference.
X86_FEATURES = frozenset(
    {
        "avx",
        "avx2",
        "fma",
        "f16c",
        "avx512f",
        "avx512bw",
        "avx512vl",
        "avx512_vnni",
        "avx512_bf16",
        "avx_vnni",
        "amx_tile",
        "amx_bf16",
        "amx_int8",
    }
)
# ARM feature names in /proc/cpuinfo -> Brasskey's names.
ARM_FEATURE_NAMES = {
    "asimd": "neon",
    "asimddp": "dotprod",
    "asimdhp": "fp16",
    "sve": "sve",
    "sve2": "sve2",
    "i8mm": "i8mm",
    "bf16": "bf16",
    "sme": "sme",
    "sme2": "sme2",
}
# macOS sysctl feature flags on Apple Silicon -> Brasskey's names.
MAC_ARM_FEATURES = {
    "hw.optional.neon": "neon",
    "hw.optional.arm.FEAT_DotProd": "dotprod",
    "hw.optional.arm.FEAT_I8MM": "i8mm",
    "hw.optional.arm.FEAT_BF16": "bf16",
    "hw.optional.arm.FEAT_FP16": "fp16",
    "hw.optional.arm.FEAT_SME": "sme",
    "hw.optional.arm.FEAT_SME2": "sme2",
}
# Intel Mac sysctl feature names (lower-cased) -> Brasskey's names.
MAC_X86_FEATURES = {
    "avx1.0": "avx",
    "avx2": "avx2",
    "fma": "fma",
    "f16c": "f16c",
    "avx512f": "avx512f",
    "avx512bw": "avx512bw",
    "avx512vl": "avx512vl",
    "avx512vnni": "avx512_vnni",
}
MAC_CPU_KEYS = (
    "machdep.cpu.brand_string",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel1.physicalcpu",
    "hw.optional.arm64",
    "sysctl.proc_translated",
    "machdep.cpu.features",
    "machdep.cpu.leaf7_features",
    *MAC_ARM_FEATURES,
)
# (implementer, part) from /proc/cpuinfo on ARM Linux -> marketing name. Unknown parts fall back
# to a descriptive string with the raw ids.
ARM_PARTS = {
    ("0x41", "0xd08"): "ARM Cortex-A72",
    ("0x41", "0xd0b"): "ARM Cortex-A76",
    ("0x41", "0xd0c"): "ARM Neoverse-N1",
    ("0x41", "0xd40"): "ARM Neoverse-V1",
    ("0x41", "0xd49"): "ARM Neoverse-N2",
    ("0x41", "0xd4f"): "ARM Neoverse-V2",
}
# ARM implementer ids -> vendor, for parts not in ARM_PARTS (e.g. Linux VMs on Apple Silicon).
ARM_IMPLEMENTERS = {
    "0x41": "ARM",
    "0x48": "HiSilicon",
    "0x4e": "NVIDIA",
    "0x61": "Apple",
    "0xc0": "Ampere",
}
# cgroup v1 reports "no limit" as a huge number close to 2**63.
CGROUP_V1_UNLIMITED = 1 << 60


class SystemStats(Protocol):
    """System counters (psutil in production, a fake in tests)."""

    def cpu_count(self, *, logical: bool) -> int | None: ...
    def memory(self) -> tuple[int, int]: ...  # (total, available) bytes
    def usable_cpus(self) -> int | None: ...  # CPUs in this process's affinity mask


class PsutilStats:
    def cpu_count(self, *, logical: bool) -> int | None:
        return psutil.cpu_count(logical=logical)

    def memory(self) -> tuple[int, int]:
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)

    def usable_cpus(self) -> int | None:
        if sys.platform == "linux":
            return len(os.sched_getaffinity(0))
        return None


def normalize_arch(machine: str) -> CpuArch:
    lowered = machine.lower()
    if lowered in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64", "armv8l"}:
        return "arm64"
    return "other"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_cpuinfo(text: str | None) -> dict[str, str]:
    """Fields of the first processor block in /proc/cpuinfo."""
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        if not line.strip():
            if fields:
                break
            continue
        key, separator, value = line.partition(":")
        if separator:
            fields.setdefault(key.strip(), value.strip())
    return fields


def _linux_model_name(fields: Mapping[str, str]) -> str | None:
    if name := fields.get("model name"):
        return name
    implementer = fields.get("CPU implementer", "").lower()
    part = fields.get("CPU part", "").lower()
    if implementer and part:
        if known := ARM_PARTS.get((implementer, part)):
            return known
        if vendor := ARM_IMPLEMENTERS.get(implementer):
            return f"{vendor} ARM CPU (part {part})"
        return f"ARM CPU (implementer {implementer}, part {part})"
    return None


def _linux_features(fields: Mapping[str, str], arch: CpuArch) -> tuple[str, ...]:
    if arch == "x86_64":
        return tuple(flag for flag in fields.get("flags", "").split() if flag in X86_FEATURES)
    if arch == "arm64":
        names = fields.get("Features", "").split()
        return tuple(ARM_FEATURE_NAMES[name] for name in names if name in ARM_FEATURE_NAMES)
    return ()


def _cgroup_v2_path(text: str | None) -> PurePosixPath | None:
    """This process's cgroup v2 path from /proc/self/cgroup (the "0::/path" line)."""
    for line in (text or "").splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            return PurePosixPath(path.strip() or "/")
    return None


def _parse_cpu_max(text: str | None) -> float | None:
    """cgroup v2 cpu.max: "max 100000" (no limit) or "<quota> <period>"."""
    parts = (text or "").split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    quota, period = parse_int(parts[0]), parse_int(parts[1])
    if quota is None or not period or quota <= 0:
        return None
    return quota / period


def _tightest(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else min(current, candidate)


def read_cgroup_limits(proc_root: Path, sys_root: Path) -> tuple[int | None, float | None]:
    """Container limits as (memory bytes, CPU cores); None means not limited.

    cgroup v2: walk from this process's cgroup up to the root and keep the tightest limit.
    cgroup v1: read the memory and cpu controllers at the root of the (namespaced) hierarchy.
    """
    base = sys_root / "fs" / "cgroup"
    own = _cgroup_v2_path(_read_text(proc_root / "self" / "cgroup"))
    if own is not None and (base / "cgroup.controllers").is_file():
        memory: float | None = None
        cpu: float | None = None
        for relative in (own, *own.parents):
            directory = base.joinpath(*relative.parts[1:])
            memory = _tightest(memory, parse_int(_read_text(directory / "memory.max")))
            cpu = _tightest(cpu, _parse_cpu_max(_read_text(directory / "cpu.max")))
        return (int(memory) if memory is not None else None), cpu

    v1_memory = parse_int(_read_text(base / "memory" / "memory.limit_in_bytes"))
    if v1_memory is not None and v1_memory >= CGROUP_V1_UNLIMITED:
        v1_memory = None
    quota = parse_int(_read_text(base / "cpu" / "cpu.cfs_quota_us"))
    period = parse_int(_read_text(base / "cpu" / "cpu.cfs_period_us"))
    v1_cpu = quota / period if quota and quota > 0 and period else None
    return v1_memory, v1_cpu


class CpuProbe:
    """Reads the CPU and system memory. Applies everywhere."""

    name = PROBE_NAME

    def __init__(
        self,
        *,
        platform_name: str | None = None,
        machine: str | None = None,
        run: CommandRunner = run_command,
        stats: SystemStats | None = None,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
    ) -> None:
        self._platform = platform_name or sys.platform
        self._machine = machine if machine is not None else platform.machine()
        self._run = run
        self._stats: SystemStats = stats if stats is not None else PsutilStats()
        self._proc_root = proc_root
        self._sys_root = sys_root

    def is_applicable(self) -> bool:
        return True

    def probe(self) -> ProbeResult:
        if self._platform == "darwin":
            return self._probe_macos()
        if self._platform.startswith("linux"):
            return self._probe_linux()
        return self._probe_generic()

    # -- shared helpers -------------------------------------------------------------------------

    def _counts(self, logical_hint: int | None, physical_hint: int | None) -> tuple[int, int]:
        logical = logical_hint or self._stats.cpu_count(logical=True) or os.cpu_count() or 1
        physical = physical_hint or self._stats.cpu_count(logical=False) or logical
        return physical, logical

    def _memory(self, cgroup_limit: int | None = None) -> MemoryInfo:
        total, available = self._stats.memory()
        limit = cgroup_limit if cgroup_limit is not None and cgroup_limit < total else None
        return MemoryInfo(total_bytes=total, available_bytes=available, cgroup_limit_bytes=limit)

    # -- platforms ------------------------------------------------------------------------------

    def _probe_macos(self) -> ProbeResult:
        values = read_sysctl(self._run, MAC_CPU_KEYS)
        apple_silicon = values.get("hw.optional.arm64") == "1"
        rosetta = values.get("sysctl.proc_translated") == "1"
        arch: CpuArch = "arm64" if apple_silicon else normalize_arch(self._machine)
        if apple_silicon:
            features = tuple(
                name for key, name in MAC_ARM_FEATURES.items() if values.get(key) == "1"
            )
        else:
            raw = f"{values.get('machdep.cpu.features', '')} "
            raw += values.get("machdep.cpu.leaf7_features", "")
            features = tuple(
                MAC_X86_FEATURES[flag] for flag in raw.lower().split() if flag in MAC_X86_FEATURES
            )
        physical, logical = self._counts(
            parse_int(values.get("hw.logicalcpu")), parse_int(values.get("hw.physicalcpu"))
        )
        cpu = CpuInfo(
            arch=arch,
            model_name=values.get("machdep.cpu.brand_string") or platform.processor() or "Unknown",
            physical_cores=physical,
            logical_cores=logical,
            performance_cores=parse_int(values.get("hw.perflevel0.physicalcpu"))
            if apple_silicon
            else None,
            efficiency_cores=parse_int(values.get("hw.perflevel1.physicalcpu"))
            if apple_silicon
            else None,
            features=features,
            rosetta=rosetta,
        )
        issues: list[ProbeIssue] = []
        if rosetta:
            issues.append(
                ProbeIssue(
                    probe=PROBE_NAME,
                    severity="warning",
                    message="This Python runs under Rosetta 2 (Intel emulation) on an Apple "
                    "Silicon Mac, which makes engines slow and MLX unusable.",
                    fix="Install a native arm64 Python (for example `uv python install 3.12`) "
                    "and reinstall Brasskey with it.",
                )
            )
        logger.debug(
            "CPU (macOS): %s, %d physical/%d logical cores, rosetta=%s",
            cpu.model_name,
            physical,
            logical,
            rosetta,
        )
        return ProbeResult(cpu=cpu, memory=self._memory(), issues=tuple(issues))

    def _probe_linux(self) -> ProbeResult:
        fields = parse_cpuinfo(_read_text(self._proc_root / "cpuinfo"))
        arch = normalize_arch(self._machine)
        physical, logical = self._counts(None, None)
        memory_limit, cpu_quota = read_cgroup_limits(self._proc_root, self._sys_root)
        usable = self._stats.usable_cpus()
        candidates = [limit for limit in (cpu_quota, usable) if limit is not None]
        cpu_limit = min(candidates) if candidates else None
        if cpu_limit is not None and cpu_limit >= logical:
            cpu_limit = None
        cpu = CpuInfo(
            arch=arch,
            model_name=_linux_model_name(fields) or platform.processor() or "Unknown CPU",
            physical_cores=physical,
            logical_cores=logical,
            features=_linux_features(fields, arch),
            cpu_limit_cores=cpu_limit,
        )
        memory = self._memory(memory_limit)
        issues: list[ProbeIssue] = []
        if memory.cgroup_limit_bytes is not None or cpu_limit is not None:
            limits: list[str] = []
            if memory.cgroup_limit_bytes is not None:
                limits.append(f"{memory.cgroup_limit_bytes / GIB:.1f} GiB of memory")
            if cpu_limit is not None:
                limits.append(f"{cpu_limit:g} CPU cores")
            issues.append(
                ProbeIssue(
                    probe=PROBE_NAME,
                    severity="info",
                    message=f"This process is limited to {' and '.join(limits)} (container or "
                    "cgroup limits); Brasskey plans within them.",
                )
            )
        logger.debug(
            "CPU (Linux): %s, %d physical/%d logical cores, cpu_limit=%s, memory_limit=%s",
            cpu.model_name,
            physical,
            logical,
            cpu_limit,
            memory.cgroup_limit_bytes,
        )
        return ProbeResult(cpu=cpu, memory=memory, issues=tuple(issues))

    def _probe_generic(self) -> ProbeResult:
        physical, logical = self._counts(None, None)
        cpu = CpuInfo(
            arch=normalize_arch(self._machine),
            model_name=platform.processor() or "Unknown CPU",
            physical_cores=physical,
            logical_cores=logical,
        )
        logger.debug("CPU (%s): %s", self._platform, cpu.model_name)
        return ProbeResult(cpu=cpu, memory=self._memory())
