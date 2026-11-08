# SPDX-License-Identifier: Apache-2.0
"""Ready-made hardware reports for planning tests (values as the real probes report them)."""

from brasskey.hardware.apple import default_gpu_memory_limit
from brasskey.hardware.types import (
    GIB,
    MIB,
    Accelerator,
    AcceleratorKind,
    CpuInfo,
    HardwareReport,
    MemoryInfo,
    OsInfo,
)

MAC = OsInfo(family="darwin", version="15.4", description="macOS 15.4")
LINUX = OsInfo(family="linux", version="6.8.0", description="Ubuntu 24.04.1 LTS (kernel 6.8.0)")


def mac(ram_gib: int, chip: str, bandwidth: float, *, available_gib: float | None = None):
    ram = ram_gib * GIB
    return HardwareReport(
        os=MAC,
        cpu=CpuInfo(arch="arm64", model_name=f"Apple {chip}", physical_cores=10, logical_cores=10),
        memory=MemoryInfo(
            total_bytes=ram,
            available_bytes=int(
                (available_gib if available_gib is not None else ram_gib / 2) * GIB
            ),
        ),
        accelerators=(
            Accelerator(
                kind=AcceleratorKind.APPLE_GPU,
                index=0,
                name=f"Apple {chip} GPU",
                memory_total_bytes=default_gpu_memory_limit(ram),
                memory_is_estimate=True,
                unified_memory=True,
                mem_bandwidth_gbs=bandwidth,
            ),
        ),
    )


def m3_pro_18gb(**kwargs):
    return mac(18, "M3 Pro", 150, **kwargs)


def m1_8gb(**kwargs):
    return mac(8, "M1", 68.25, **kwargs)


def nvidia(
    name: str,
    total_mib: int,
    cc: tuple[int, int],
    bandwidth: float,
    *,
    count: int = 1,
    used_mib: int = 500,
    ram_gib: int = 256,
):
    gpus = tuple(
        Accelerator(
            kind=AcceleratorKind.NVIDIA_GPU,
            index=index,
            name=name,
            memory_total_bytes=total_mib * MIB,
            memory_free_bytes=(total_mib - used_mib) * MIB,
            compute_capability=cc,
            driver_version="560.35.03",
            cuda_driver_version="12.6",
            mem_bandwidth_gbs=bandwidth,
        )
        for index in range(count)
    )
    return HardwareReport(
        os=LINUX,
        cpu=CpuInfo(arch="x86_64", model_name="AMD EPYC 7R32", physical_cores=48, logical_cores=96),
        memory=MemoryInfo(total_bytes=ram_gib * GIB, available_bytes=ram_gib * GIB // 2),
        accelerators=gpus,
    )


def h100(count: int = 1, **kwargs):
    return nvidia("NVIDIA H100 80GB HBM3", 81559, (9, 0), 3352, count=count, **kwargs)


def l4(count: int = 1, **kwargs):
    return nvidia("NVIDIA L4", 23034, (8, 9), 300, count=count, **kwargs)


def t4(count: int = 1, **kwargs):
    return nvidia("Tesla T4", 15360, (7, 5), 320, count=count, **kwargs)


def cpu_only(ram_gib: int, *, cgroup_gib: int | None = None, os: OsInfo = LINUX):
    return HardwareReport(
        os=os,
        cpu=CpuInfo(arch="x86_64", model_name="AMD EPYC 7R32", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(
            total_bytes=ram_gib * GIB,
            available_bytes=ram_gib * GIB * 3 // 4,
            cgroup_limit_bytes=cgroup_gib * GIB if cgroup_gib else None,
        ),
    )
