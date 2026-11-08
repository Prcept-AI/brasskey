# SPDX-License-Identifier: Apache-2.0
"""Hardware report types.

Privacy (CONTRIBUTING.md): reports end up in lock files and bug reports, so
they never contain identifying data such as GPU UUIDs, serial numbers, MAC addresses or hostnames.
"""

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GIB = 1024**3
MIB = 1024**2

CpuArch = Literal["x86_64", "arm64", "other"]
OsFamily = Literal["darwin", "linux", "windows", "other"]
Severity = Literal["info", "warning"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcceleratorKind(StrEnum):
    NVIDIA_GPU = "nvidia_gpu"
    APPLE_GPU = "apple_gpu"


class Accelerator(_Frozen):
    """One GPU (or other accelerator) Brasskey could run a model on."""

    kind: AcceleratorKind
    index: int = Field(ge=0, description="NVML index for NVIDIA GPUs; 0 for the Apple GPU.")
    name: str
    memory_total_bytes: int = Field(
        ge=0,
        description="Memory the accelerator can use. Apple: the GPU working-set limit of "
        "unified memory, not total RAM.",
    )
    memory_free_bytes: int | None = Field(
        default=None, ge=0, description="Free right now (NVIDIA); None for unified memory."
    )
    memory_is_estimate: bool = Field(
        default=False, description="True when memory_total_bytes comes from a policy estimate."
    )
    unified_memory: bool = False
    compute_capability: tuple[int, int] | None = None
    driver_version: str | None = None
    cuda_driver_version: str | None = Field(
        default=None, description="Highest CUDA version the driver supports, e.g. '12.8'."
    )
    mig_enabled: bool = False
    gpu_cores: int | None = Field(default=None, ge=1)
    mem_bandwidth_gbs: float | None = Field(
        default=None,
        gt=0,
        description="Theoretical peak memory bandwidth in GB/s (NVML bus width x memory clock, "
        "or Apple's published spec). Only used for labeled estimates.",
    )
    notes: tuple[str, ...] = ()


class CpuInfo(_Frozen):
    """The CPU, as the hardware reports it (the real architecture even under Rosetta)."""

    arch: CpuArch
    model_name: str
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    performance_cores: int | None = Field(default=None, ge=0)
    efficiency_cores: int | None = Field(default=None, ge=0)
    features: tuple[str, ...] = Field(
        default=(), description="Inference-relevant CPU features, lower case, sorted."
    )
    cpu_limit_cores: float | None = Field(
        default=None,
        gt=0,
        description="Cores this process may actually use when a container or cpuset limits it "
        "below logical_cores.",
    )
    rosetta: bool = Field(default=False, description="This Python runs under Rosetta 2.")

    @field_validator("features")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({feature.lower() for feature in value}))


class MemoryInfo(_Frozen):
    """System memory (RAM)."""

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    cgroup_limit_bytes: int | None = Field(
        default=None, ge=0, description="Container memory limit when lower than total RAM."
    )

    @property
    def usable_total_bytes(self) -> int:
        """Total memory this process can use: RAM, or the container limit if lower."""
        if self.cgroup_limit_bytes is None:
            return self.total_bytes
        return min(self.total_bytes, self.cgroup_limit_bytes)


class ProbeIssue(_Frozen):
    """Something the user should know about the hardware, in plain language with a fix."""

    probe: str
    severity: Severity
    message: str
    fix: str | None = None


class ProbeResult(_Frozen):
    """What one probe found. Probes fill in only the parts they know about."""

    accelerators: tuple[Accelerator, ...] = ()
    cpu: CpuInfo | None = None
    memory: MemoryInfo | None = None
    issues: tuple[ProbeIssue, ...] = ()


class OsInfo(_Frozen):
    family: OsFamily
    version: str = Field(description="macOS version, or the kernel release on Linux.")
    description: str = Field(description="Human label, e.g. 'macOS 15.4' or 'Ubuntu 24.04 LTS'.")


class HardwareReport(_Frozen):
    """Everything the hardware probe found on this machine."""

    schema_version: Literal[1] = 1
    os: OsInfo
    cpu: CpuInfo
    memory: MemoryInfo
    accelerators: tuple[Accelerator, ...] = ()
    issues: tuple[ProbeIssue, ...] = ()

    def accelerators_of(self, kind: AcceleratorKind) -> tuple[Accelerator, ...]:
        return tuple(acc for acc in self.accelerators if acc.kind == kind)

    def fingerprint(self) -> str:
        """Stable, non-identifying hash of the hardware, for brasskey.lock and registry records.

        It covers what changes performance (CPU model, cores, features, RAM, accelerators) and
        ignores what changes from minute to minute (free memory, issues) or with updates (OS and
        driver versions, recorded separately). Memory is rounded to whole GiB.
        """
        payload = {
            "os": self.os.family,
            "cpu": {
                "arch": self.cpu.arch,
                "model": self.cpu.model_name,
                "physical_cores": self.cpu.physical_cores,
                "logical_cores": self.cpu.logical_cores,
                "features": list(self.cpu.features),
            },
            "memory_gib": round(self.memory.total_bytes / GIB),
            "accelerators": [
                {
                    "kind": acc.kind.value,
                    "name": acc.name,
                    "memory_gib": round(acc.memory_total_bytes / GIB),
                    "compute_capability": list(acc.compute_capability or ()),
                    "gpu_cores": acc.gpu_cores,
                }
                for acc in self.accelerators
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
