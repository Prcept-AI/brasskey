# SPDX-License-Identifier: Apache-2.0
"""Which engines Brasskey can run in this build, what is installed, and what engines measure.

Every v0.1 engine has an adapter: llama.cpp (milestone d), MLX (e), vLLM and SGLang (g).
The planner only chooses engines that have an adapter, so a dry run never proposes something
``brasskey run`` can't start.
"""

import logging
import time
from collections.abc import Mapping

from brasskey.engines.base import EngineAdapter, EngineInstall
from brasskey.engines.llamacpp import LlamaCppAdapter
from brasskey.engines.mlx import MlxAdapter
from brasskey.engines.sglang import SglangAdapter
from brasskey.engines.types import EngineId
from brasskey.engines.vllm import VllmAdapter
from brasskey.hardware.types import AcceleratorKind, HardwareReport

logger = logging.getLogger(__name__)

_ADAPTERS: dict[EngineId, type[EngineAdapter]] = {
    EngineId.LLAMACPP: LlamaCppAdapter,
    EngineId.MLX: MlxAdapter,
    EngineId.VLLM: VllmAdapter,
    EngineId.SGLANG: SglangAdapter,
}

METAL_NOTE = "GPU memory limit reported by Metal (measured by an installed engine)."


def runnable_engines() -> tuple[EngineId, ...]:
    """Engines this build can start."""
    return tuple(_ADAPTERS)


def adapter_for(engine: EngineId) -> EngineAdapter | None:
    cls = _ADAPTERS.get(engine)
    return cls() if cls is not None else None


def detect_engines(
    adapters: Mapping[EngineId, EngineAdapter] | None = None,
) -> dict[EngineId, EngineInstall | None]:
    """Installed engines (None when an engine isn't installed)."""
    chosen = adapters if adapters is not None else {e: cls() for e, cls in _ADAPTERS.items()}
    found: dict[EngineId, EngineInstall | None] = {}
    for engine, adapter in chosen.items():
        started = time.perf_counter()
        install = adapter.detect()
        found[engine] = install
        logger.debug(
            "Detect %s: %s in %.0f ms",
            engine.value,
            install.version if install is not None else "not installed",
            (time.perf_counter() - started) * 1000,
        )
    return found


def refine_hardware(
    hw: HardwareReport, installs: Mapping[EngineId, EngineInstall | None]
) -> HardwareReport:
    """Replace estimates with what installed engines measured.

    Today: the Apple GPU memory limit. llama.cpp reports Metal's ``recommendedMaxWorkingSetSize``
    with ``--list-devices``, which turns the macOS-default estimate into a measurement.
    """
    apple = hw.accelerators_of(AcceleratorKind.APPLE_GPU)
    if not apple or not apple[0].memory_is_estimate:
        return hw
    metal = next(
        (
            device
            for install in installs.values()
            if install is not None
            for device in install.devices
            if device.backend == "metal" and device.memory_total_bytes
        ),
        None,
    )
    if metal is None or metal.memory_total_bytes is None:
        return hw
    measured = apple[0].model_copy(
        update={
            "memory_total_bytes": metal.memory_total_bytes,
            "memory_is_estimate": False,
            "notes": (*(n for n in apple[0].notes if "estimate" not in n), METAL_NOTE),
        }
    )
    logger.debug("Apple GPU limit measured by Metal: %d bytes", metal.memory_total_bytes)
    accelerators = tuple(measured if acc is apple[0] else acc for acc in hw.accelerators)
    return hw.model_copy(update={"accelerators": accelerators})
