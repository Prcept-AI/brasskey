# SPDX-License-Identifier: Apache-2.0
"""NVIDIA GPU probe through NVML (read-only).

NVML comes from NVIDIA's ``nvidia-ml-py`` bindings (BSD-3-Clause), which load the driver's
``libnvidia-ml`` at runtime; that is a driver library, not an inference engine. The probe talks to
NVML through the small :class:`NvmlApi` protocol so tests can use a fake.

Handled here:
- no driver or no GPU: silent when there is no NVIDIA hardware, a plain fix when there is;
- driver/library version mismatch (usually after a driver update without a reboot), permissions;
- per-GPU failures (a lost GPU is skipped, the others are still reported);
- MIG mode, memory already used by other processes;
- ``CUDA_VISIBLE_DEVICES``: engines only see the GPUs it selects, so Brasskey plans with those.

GPU UUIDs are read only to match ``CUDA_VISIBLE_DEVICES`` and are never stored or logged.
"""

import importlib
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

from brasskey.hardware.types import (
    GIB,
    Accelerator,
    AcceleratorKind,
    ProbeIssue,
    ProbeResult,
    Severity,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PROBE_NAME = "nvidia"
NVIDIA_PCI_VENDOR = "0x10de"
NVIDIA_SMI_HINT = "check that `nvidia-smi` lists your GPUs"
# Memory used by other processes above this share (and above 1 GiB) is worth a warning.
BUSY_MEMORY_SHARE = 0.10


class NvmlErrorKind(StrEnum):
    LIBRARY_NOT_FOUND = "library_not_found"
    DRIVER_NOT_LOADED = "driver_not_loaded"
    NO_PERMISSION = "no_permission"
    NOT_SUPPORTED = "not_supported"
    VERSION_MISMATCH = "version_mismatch"
    GPU_LOST = "gpu_lost"
    OTHER = "other"


class NvmlError(Exception):
    def __init__(self, kind: NvmlErrorKind, detail: str = "") -> None:
        super().__init__(f"{kind.value}: {detail}" if detail else kind.value)
        self.kind = kind
        self.detail = detail


class NvmlApi(Protocol):
    """The NVML calls the probe uses. Every method raises :class:`NvmlError` on failure."""

    def init(self) -> None: ...
    def shutdown(self) -> None: ...
    def driver_version(self) -> str: ...
    def cuda_driver_version(self) -> int: ...  # e.g. 12080 for CUDA 12.8
    def device_count(self) -> int: ...
    def device_handle(self, index: int) -> object: ...
    def device_name(self, handle: object) -> str: ...
    def device_uuid(self, handle: object) -> str: ...
    def device_memory(self, handle: object) -> tuple[int, int]: ...  # (total, free) bytes
    def device_compute_capability(self, handle: object) -> tuple[int, int]: ...
    def device_memory_bus_width(self, handle: object) -> int: ...  # bits
    def device_max_memory_clock_mhz(self, handle: object) -> int: ...
    def device_mig_enabled(self, handle: object) -> bool: ...


class RealNvml:
    """:class:`NvmlApi` backed by ``pynvml`` (from nvidia-ml-py). Imported lazily."""

    def __init__(self) -> None:
        self._module: Any = None

    def _lib(self) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module("pynvml")
            except ImportError as error:  # pragma: no cover - it's a declared dependency
                raise NvmlError(NvmlErrorKind.LIBRARY_NOT_FOUND, "pynvml not installed") from error
        return self._module

    def _kind(self, code: object) -> NvmlErrorKind:
        lib = self._lib()
        mapping = {
            "NVML_ERROR_LIBRARY_NOT_FOUND": NvmlErrorKind.LIBRARY_NOT_FOUND,
            "NVML_ERROR_DRIVER_NOT_LOADED": NvmlErrorKind.DRIVER_NOT_LOADED,
            "NVML_ERROR_NO_PERMISSION": NvmlErrorKind.NO_PERMISSION,
            "NVML_ERROR_NOT_SUPPORTED": NvmlErrorKind.NOT_SUPPORTED,
            "NVML_ERROR_FUNCTION_NOT_FOUND": NvmlErrorKind.NOT_SUPPORTED,
            "NVML_ERROR_LIB_RM_VERSION_MISMATCH": NvmlErrorKind.VERSION_MISMATCH,
            "NVML_ERROR_GPU_IS_LOST": NvmlErrorKind.GPU_LOST,
            "NVML_ERROR_RESET_REQUIRED": NvmlErrorKind.GPU_LOST,
        }
        for constant, kind in mapping.items():
            if getattr(lib, constant, None) == code:
                return kind
        return NvmlErrorKind.OTHER

    def _call(self, function: str, *args: object) -> Any:
        lib = self._lib()
        try:
            return getattr(lib, function)(*args)
        except lib.NVMLError as error:
            raise NvmlError(self._kind(getattr(error, "value", None)), str(error)) from error

    def init(self) -> None:
        self._call("nvmlInit")

    def shutdown(self) -> None:
        self._call("nvmlShutdown")

    def driver_version(self) -> str:
        return str(self._call("nvmlSystemGetDriverVersion"))

    def cuda_driver_version(self) -> int:
        return int(self._call("nvmlSystemGetCudaDriverVersion"))

    def device_count(self) -> int:
        return int(self._call("nvmlDeviceGetCount"))

    def device_handle(self, index: int) -> object:
        return self._call("nvmlDeviceGetHandleByIndex", index)

    def device_name(self, handle: object) -> str:
        return str(self._call("nvmlDeviceGetName", handle))

    def device_uuid(self, handle: object) -> str:
        return str(self._call("nvmlDeviceGetUUID", handle))

    def device_memory(self, handle: object) -> tuple[int, int]:
        info = self._call("nvmlDeviceGetMemoryInfo", handle)
        return int(info.total), int(info.free)

    def device_compute_capability(self, handle: object) -> tuple[int, int]:
        major, minor = self._call("nvmlDeviceGetCudaComputeCapability", handle)
        return int(major), int(minor)

    def device_memory_bus_width(self, handle: object) -> int:
        return int(self._call("nvmlDeviceGetMemoryBusWidth", handle))

    def device_max_memory_clock_mhz(self, handle: object) -> int:
        return int(self._call("nvmlDeviceGetMaxClockInfo", handle, self._lib().NVML_CLOCK_MEM))

    def device_mig_enabled(self, handle: object) -> bool:
        current, _pending = self._call("nvmlDeviceGetMigMode", handle)
        return bool(current == self._lib().NVML_DEVICE_MIG_ENABLE)


def format_cuda_version(raw: int) -> str:
    """NVML's integer CUDA version (12080) as "12.8"."""
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def peak_bandwidth_gbs(bus_width_bits: int, memory_clock_mhz: int) -> float | None:
    """Theoretical peak bandwidth: bus bytes x memory clock x 2 (double data rate).

    Matches the published figures, e.g. H100 SXM (5120-bit, 2619 MHz) = 3352 GB/s and
    RTX 4090 (384-bit, 10501 MHz) = 1008 GB/s.
    """
    if bus_width_bits <= 0 or memory_clock_mhz <= 0:
        return None
    return round(bus_width_bits / 8 * memory_clock_mhz * 2 / 1000, 1)


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """What ``CUDA_VISIBLE_DEVICES`` selects (CUDA stops at the first invalid entry)."""

    indices: tuple[int, ...]
    uuid_prefixes: tuple[str, ...]  # matched in memory only; never stored or printed
    has_mig_entries: bool

    def selects(self, index: int, uuid: str | None) -> bool:
        if index in self.indices:
            return True
        return uuid is not None and any(uuid.startswith(prefix) for prefix in self.uuid_prefixes)


def parse_visible_devices(value: str | None) -> DeviceSelection | None:
    """Parse ``CUDA_VISIBLE_DEVICES``; None when unset (all GPUs visible)."""
    if value is None:
        return None
    indices: list[int] = []
    prefixes: list[str] = []
    has_mig = False
    for entry in (part.strip() for part in value.split(",")):
        if entry.isdigit():
            indices.append(int(entry))
        elif entry.startswith("GPU-"):
            prefixes.append(entry)
        elif entry.startswith("MIG-"):
            has_mig = True
        else:
            break  # e.g. "-1" or "": CUDA ignores this entry and everything after it
    return DeviceSelection(tuple(indices), tuple(prefixes), has_mig)


def nvidia_pci_present(sys_root: Path) -> bool:
    """True if an NVIDIA display/3D controller is on the PCI bus (Linux sysfs)."""
    devices = sys_root / "bus" / "pci" / "devices"
    try:
        entries = list(devices.iterdir())
    except OSError:
        return False
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text().strip().lower()
            device_class = (entry / "class").read_text().strip().lower()
        except OSError:
            continue
        if vendor == NVIDIA_PCI_VENDOR and device_class.startswith("0x03"):
            return True
    return False


def _issue(severity: Severity, message: str, fix: str | None = None) -> ProbeIssue:
    return ProbeIssue(probe=PROBE_NAME, severity=severity, message=message, fix=fix)


class NvmlProbe:
    """Reports NVIDIA GPUs on Linux (and Windows/WSL)."""

    name = PROBE_NAME

    def __init__(
        self,
        nvml: NvmlApi | None = None,
        *,
        platform_name: str | None = None,
        env: Mapping[str, str] | None = None,
        sys_root: Path = Path("/sys"),
        pci_check: Callable[[Path], bool] = nvidia_pci_present,
    ) -> None:
        self._nvml = nvml
        self._platform = platform_name or sys.platform
        self._env = env if env is not None else os.environ
        self._sys_root = sys_root
        self._pci_check = pci_check

    def is_applicable(self) -> bool:
        # macOS hasn't supported NVIDIA GPUs since 10.13.
        return self._platform.startswith("linux") or self._platform == "win32"

    def probe(self) -> ProbeResult:
        nvml = self._nvml if self._nvml is not None else RealNvml()
        try:
            nvml.init()
        except NvmlError as error:
            logger.debug("NVML init failed: %s", error.kind.value)
            return ProbeResult(issues=self._init_issues(error))
        try:
            return self._read_devices(nvml)
        except NvmlError as error:
            logger.debug("NVML failed while listing GPUs: %s", error.kind.value)
            return ProbeResult(
                issues=(
                    _issue(
                        "warning",
                        f"Couldn't list NVIDIA GPUs ({error.kind.value}).",
                        f"Reboot if the driver was just updated, and {NVIDIA_SMI_HINT}.",
                    ),
                )
            )
        finally:
            try:
                nvml.shutdown()
            except NvmlError:
                logger.debug("NVML shutdown failed (ignored)")

    def _init_issues(self, error: NvmlError) -> tuple[ProbeIssue, ...]:
        kind = error.kind
        if kind == NvmlErrorKind.LIBRARY_NOT_FOUND:
            if not self._pci_check(self._sys_root):
                return ()  # no NVIDIA hardware, nothing to report
            return (
                _issue(
                    "warning",
                    "An NVIDIA GPU is installed but its driver library (libnvidia-ml) wasn't "
                    "found, so Brasskey can't use it.",
                    f"Install the NVIDIA driver, reboot, and {NVIDIA_SMI_HINT}.",
                ),
            )
        if kind == NvmlErrorKind.DRIVER_NOT_LOADED:
            return (
                _issue(
                    "warning",
                    "The NVIDIA driver isn't loaded, so Brasskey can't use the GPU.",
                    f"Reboot (or run `sudo modprobe nvidia`), then {NVIDIA_SMI_HINT}.",
                ),
            )
        if kind == NvmlErrorKind.VERSION_MISMATCH:
            return (
                _issue(
                    "warning",
                    "The NVIDIA driver and its library have different versions, which usually "
                    "happens after a driver update.",
                    "Reboot the machine.",
                ),
            )
        if kind == NvmlErrorKind.NO_PERMISSION:
            return (
                _issue(
                    "warning",
                    "Brasskey isn't allowed to read the NVIDIA GPUs.",
                    "Run as a user with access to /dev/nvidia* (in Docker, pass `--gpus all`).",
                ),
            )
        return (
            _issue(
                "warning",
                f"Couldn't start NVML to read NVIDIA GPUs ({error.detail or kind.value}).",
                f"Check the NVIDIA driver installation; {NVIDIA_SMI_HINT}.",
            ),
        )

    def _read_devices(self, nvml: NvmlApi) -> ProbeResult:
        issues: list[ProbeIssue] = []
        driver = _optional(nvml.driver_version)
        cuda_raw = _optional(nvml.cuda_driver_version)
        cuda = format_cuda_version(cuda_raw) if cuda_raw else None
        selection = parse_visible_devices(self._env.get("CUDA_VISIBLE_DEVICES"))
        if selection is not None and selection.has_mig_entries:
            issues.append(
                _issue(
                    "info",
                    "CUDA_VISIBLE_DEVICES selects MIG instances; Brasskey v0.1 reports whole "
                    "GPUs and doesn't apply that selection.",
                )
            )
            selection = None

        count = nvml.device_count()
        found: list[Accelerator] = []
        names: list[str] = []
        for index in range(count):
            try:
                handle = nvml.device_handle(index)
                name = nvml.device_name(handle)
                total, free = nvml.device_memory(handle)
                capability = nvml.device_compute_capability(handle)
            except NvmlError as error:
                logger.debug("GPU %d unreadable: %s", index, error.kind.value)
                issues.append(
                    _issue(
                        "warning",
                        f"NVIDIA GPU {index} couldn't be read ({error.kind.value}) and is skipped.",
                        f"Reboot if the GPU fell off the bus, and {NVIDIA_SMI_HINT}.",
                    )
                )
                continue
            names.append(name)

            if selection is not None:
                uuid = _optional(partial(nvml.device_uuid, handle))
                if not selection.selects(index, uuid):
                    logger.debug("GPU %d hidden by CUDA_VISIBLE_DEVICES", index)
                    continue

            bus_width = _optional(partial(nvml.device_memory_bus_width, handle))
            clock = _optional(partial(nvml.device_max_memory_clock_mhz, handle))
            bandwidth = peak_bandwidth_gbs(bus_width, clock) if bus_width and clock else None
            mig = bool(_optional(partial(nvml.device_mig_enabled, handle)))

            notes: list[str] = []
            if bandwidth is not None:
                notes.append("Peak bandwidth from NVML bus width x memory clock.")
            if mig:
                notes.append("MIG is enabled: engines see MIG instances, not the whole GPU.")
                issues.append(
                    _issue(
                        "warning",
                        f"NVIDIA GPU {index} ({name}) has MIG enabled; Brasskey v0.1 plans for "
                        "whole GPUs.",
                        f"Disable MIG (`sudo nvidia-smi -i {index} -mig 0`) or plan sizes by hand.",
                    )
                )
            used = max(total - free, 0)
            if used > max(GIB, BUSY_MEMORY_SHARE * total):
                issues.append(
                    _issue(
                        "warning",
                        f"NVIDIA GPU {index} ({name}) has {used / GIB:.1f} of "
                        f"{total / GIB:.1f} GiB in use by other processes.",
                        "Stop the other GPU processes (see `nvidia-smi`) or pick a smaller model.",
                    )
                )
            found.append(
                Accelerator(
                    kind=AcceleratorKind.NVIDIA_GPU,
                    index=index,
                    name=name,
                    memory_total_bytes=total,
                    memory_free_bytes=free,
                    compute_capability=capability,
                    driver_version=driver,
                    cuda_driver_version=cuda,
                    mig_enabled=mig,
                    mem_bandwidth_gbs=bandwidth,
                    notes=tuple(notes),
                )
            )
            logger.debug(
                "GPU %d: %s, %.1f GiB (%.1f free), cc %s, %s GB/s, mig=%s",
                index,
                name,
                total / GIB,
                free / GIB,
                capability,
                bandwidth,
                mig,
            )

        if selection is not None:
            issues.extend(self._selection_issues(selection, found, count, names))
        return ProbeResult(accelerators=tuple(found), issues=tuple(issues))

    def _selection_issues(
        self,
        selection: DeviceSelection,
        visible: list[Accelerator],
        count: int,
        names: list[str],
    ) -> list[ProbeIssue]:
        issues: list[ProbeIssue] = []
        shown = ", ".join(str(acc.index) for acc in visible) or "none"
        issues.append(
            _issue(
                "info",
                # The raw value is not echoed: it may contain GPU UUIDs.
                f"CUDA_VISIBLE_DEVICES limits Brasskey to NVIDIA GPUs {shown} "
                f"({len(visible)} of {count}).",
            )
        )
        reordered = self._env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        if selection.indices and len(set(names)) > 1 and reordered:
            issues.append(
                _issue(
                    "warning",
                    "These GPUs are different models, so CUDA may number them differently from "
                    "NVML and CUDA_VISIBLE_DEVICES may pick other GPUs than intended.",
                    "Set CUDA_DEVICE_ORDER=PCI_BUS_ID, or select GPUs by UUID.",
                )
            )
        return issues


def _optional(read: Callable[[], T]) -> T | None:
    """Value of an optional NVML query; None when it isn't supported or fails."""
    try:
        return read()
    except NvmlError as error:
        logger.debug("Optional NVML query unavailable: %s", error.kind.value)
        return None
