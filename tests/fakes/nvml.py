# SPDX-License-Identifier: Apache-2.0
"""A fake NVML with real GPU specifications, implementing ``brasskey.hardware.nvidia.NvmlApi``.

Specs (memory, compute capability, bus width, max memory clock) match what NVML reports on the
real cards, so bandwidth figures can be checked against the vendors' published numbers.
"""

from dataclasses import dataclass, replace

from brasskey.hardware.nvidia import NvmlError, NvmlErrorKind
from brasskey.hardware.types import MIB


@dataclass(frozen=True)
class FakeGpu:
    name: str
    total_mib: int
    compute_capability: tuple[int, int]
    bus_width: int | None  # None: NVML says "not supported"
    memory_clock_mhz: int | None
    mig: bool | None = None  # None: MIG not supported on this GPU
    used_mib: int = 500
    uuid: str = "GPU-00000000-0000-0000-0000-000000000000"
    error: NvmlErrorKind | None = None  # raised when the device is read


H100_SXM = FakeGpu("NVIDIA H100 80GB HBM3", 81559, (9, 0), 5120, 2619, mig=False)
A100_40GB = FakeGpu("NVIDIA A100-SXM4-40GB", 40960, (8, 0), 5120, 1215, mig=False)
RTX_4090 = FakeGpu("NVIDIA GeForce RTX 4090", 24564, (8, 9), 384, 10501)
L4 = FakeGpu("NVIDIA L4", 23034, (8, 9), 192, 6251)
T4 = FakeGpu("Tesla T4", 15360, (7, 5), 256, 5001)


def fleet(gpu: FakeGpu, count: int) -> list[FakeGpu]:
    """``count`` copies of a GPU with distinct UUIDs."""
    return [
        replace(gpu, uuid=f"GPU-{index:08x}-aaaa-bbbb-cccc-000000000000") for index in range(count)
    ]


class FakeNvml:
    def __init__(
        self,
        gpus: list[FakeGpu],
        *,
        driver: str = "560.35.03",
        cuda: int = 12060,
        init_error: NvmlErrorKind | None = None,
        count_error: NvmlErrorKind | None = None,
    ) -> None:
        self.gpus = gpus
        self.driver = driver
        self.cuda = cuda
        self.init_error = init_error
        self.count_error = count_error
        self.shut_down = False

    def _gpu(self, handle: object) -> FakeGpu:
        assert isinstance(handle, int)
        gpu = self.gpus[handle]
        if gpu.error is not None:
            raise NvmlError(gpu.error, "fake device error")
        return gpu

    def init(self) -> None:
        if self.init_error is not None:
            raise NvmlError(self.init_error, "fake init error")

    def shutdown(self) -> None:
        self.shut_down = True

    def driver_version(self) -> str:
        return self.driver

    def cuda_driver_version(self) -> int:
        return self.cuda

    def device_count(self) -> int:
        if self.count_error is not None:
            raise NvmlError(self.count_error, "fake count error")
        return len(self.gpus)

    def device_handle(self, index: int) -> object:
        return index

    def device_name(self, handle: object) -> str:
        return self._gpu(handle).name

    def device_uuid(self, handle: object) -> str:
        return self._gpu(handle).uuid

    def device_memory(self, handle: object) -> tuple[int, int]:
        gpu = self._gpu(handle)
        return gpu.total_mib * MIB, (gpu.total_mib - gpu.used_mib) * MIB

    def device_compute_capability(self, handle: object) -> tuple[int, int]:
        return self._gpu(handle).compute_capability

    def device_memory_bus_width(self, handle: object) -> int:
        value = self._gpu(handle).bus_width
        if value is None:
            raise NvmlError(NvmlErrorKind.NOT_SUPPORTED)
        return value

    def device_max_memory_clock_mhz(self, handle: object) -> int:
        value = self._gpu(handle).memory_clock_mhz
        if value is None:
            raise NvmlError(NvmlErrorKind.NOT_SUPPORTED)
        return value

    def device_mig_enabled(self, handle: object) -> bool:
        value = self._gpu(handle).mig
        if value is None:
            raise NvmlError(NvmlErrorKind.NOT_SUPPORTED)
        return value


__all__ = ["A100_40GB", "H100_SXM", "L4", "RTX_4090", "T4", "FakeGpu", "FakeNvml", "fleet"]
