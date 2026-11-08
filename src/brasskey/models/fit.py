# SPDX-License-Identifier: Apache-2.0
"""Will a model fit? Memory budget per device, memory needed, and a decode-speed ceiling.

Needed memory per device = weights / devices + KV cache share + engine overhead, where
- weights are the measured size of the weight files (what the engine holds in memory),
- the KV cache is the architecture's bytes per token x context length x concurrent sequences,
- overhead is the engine profile's fixed cost plus a share of the weights.

Everything here is an estimate and labeled as one; measurements replace it later
(CONTRIBUTING.md: "measure, don't guess"). See docs/decisions/0004-model-fit-and-planning.md.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brasskey.engines.capabilities import device_pool, memory_profile, unused_gpu_warning
from brasskey.engines.types import DevicePool, EngineId, EngineMemoryProfile
from brasskey.hardware.types import GIB, AcceleratorKind, HardwareReport
from brasskey.models.types import CatalogModel, ModelArchitecture, ModelArtifact

logger = logging.getLogger(__name__)

HEADROOM_TARGET = 0.25  # a comfortable fit leaves a quarter of the budget free
CPU_RESERVED_MIN = GIB  # RAM kept for the OS and other programs...
CPU_RESERVED_SHARE = 0.15  # ...or this share of RAM, whichever is larger
CONTAINER_RESERVED_SHARE = 0.10  # in a container the OS lives outside the limit
MIN_CONTEXT = 256
TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MemoryBudget(_Frozen):
    pool: DevicePool
    device_count: int = Field(ge=1)
    per_device_bytes: int = Field(ge=0, description="Usable by the engine on each device.")
    available_now_bytes: int | None = Field(
        default=None, description="Free right now (RAM, or the least-free GPU)."
    )
    is_estimate: bool
    explanation: str


class FitEstimate(_Frozen):
    model_id: str
    engine: EngineId
    artifact: ModelArtifact
    context_length: int
    concurrency: int
    devices_used: int
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    required_bytes: int
    required_per_device_bytes: int
    budget: MemoryBudget
    fits: bool
    headroom_ratio: float
    max_context_that_fits: int
    est_decode_tokens_per_s: float | None = Field(
        default=None, description="Upper bound: memory bandwidth / weight bytes read per token."
    )
    is_estimate: Literal[True] = True
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def comfortable(self) -> bool:
        return self.fits and self.headroom_ratio >= HEADROOM_TARGET


def _gib(size: int) -> str:
    return f"{size / GIB:.1f} GiB"


def memory_budget(
    engine: EngineId,
    hw: HardwareReport,
    profile: EngineMemoryProfile,
    *,
    pool: DevicePool | None = None,
) -> MemoryBudget:
    """How much memory ``engine`` may use on this machine (in ``pool``, default: its usual one)."""
    pool = pool if pool is not None else device_pool(engine, hw)
    if pool == DevicePool.APPLE_GPU:
        gpu = hw.accelerators_of(AcceleratorKind.APPLE_GPU)[0]
        about = "about " if gpu.memory_is_estimate else ""
        return MemoryBudget(
            pool=pool,
            device_count=1,
            per_device_bytes=int(gpu.memory_total_bytes * profile.max_fraction),
            available_now_bytes=hw.memory.available_bytes,
            is_estimate=gpu.memory_is_estimate,
            explanation=f"the GPU can use {about}{_gib(gpu.memory_total_bytes)} of "
            f"{_gib(hw.memory.total_bytes)} unified memory",
        )
    if pool == DevicePool.NVIDIA_GPU:
        gpus = hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
        smallest = min(gpu.memory_total_bytes for gpu in gpus)
        free = [gpu.memory_free_bytes for gpu in gpus if gpu.memory_free_bytes is not None]
        count = len(gpus)
        return MemoryBudget(
            pool=pool,
            device_count=count,
            per_device_bytes=int(smallest * profile.max_fraction),
            available_now_bytes=min(free) if free else None,
            is_estimate=False,
            explanation=f"{engine.display_name} may use {profile.max_fraction:.0%} of each "
            f"GPU's {_gib(smallest)} ({count} GPU{'s' if count > 1 else ''})",
        )
    usable = hw.memory.usable_total_bytes
    if hw.memory.cgroup_limit_bytes is not None:
        reserved = int(usable * CONTAINER_RESERVED_SHARE)
        explanation = f"{_gib(usable)} container limit minus {_gib(reserved)} headroom"
    else:
        reserved = max(CPU_RESERVED_MIN, int(usable * CPU_RESERVED_SHARE))
        explanation = f"{_gib(usable)} RAM minus {_gib(reserved)} kept for the system"
    return MemoryBudget(
        pool=pool,
        device_count=1,
        per_device_bytes=max(usable - reserved, 0),
        available_now_bytes=hw.memory.available_bytes,
        is_estimate=False,
        explanation=explanation,
    )


def _device_counts(engine: EngineId, budget: MemoryBudget, arch: ModelArchitecture) -> list[int]:
    """Ways to split the model across devices, fewest devices first."""
    if budget.pool != DevicePool.NVIDIA_GPU:
        return [1]
    if engine in (EngineId.VLLM, EngineId.SGLANG):
        # Tensor parallelism: attention heads must divide evenly.
        sizes = [
            size
            for size in TENSOR_PARALLEL_SIZES
            if size <= budget.device_count
            and arch.num_attention_heads % size == 0
            and (arch.num_kv_heads % size == 0 or size % arch.num_kv_heads == 0)
        ]
        return sizes or [1]
    return list(range(1, budget.device_count + 1))  # llama.cpp splits layers across GPUs


def _kv_per_device(kv_total: int, devices: int, arch: ModelArchitecture) -> int:
    if devices == 1:
        return kv_total
    if arch.uses_mla:
        return kv_total  # the MLA latent is replicated on every device
    shards = min(devices, arch.num_kv_heads)  # beyond the KV head count, heads are replicated
    return -(-kv_total // shards)


def _per_device_need(
    weights: int, kv_total: int, devices: int, profile: EngineMemoryProfile, arch: ModelArchitecture
) -> int:
    weights_share = -(-weights // devices)
    overhead = profile.overhead_bytes + int(profile.overhead_fraction * weights_share)
    return weights_share + _kv_per_device(kv_total, devices, arch) + overhead


def _max_context(
    arch: ModelArchitecture,
    weights: int,
    devices: int,
    concurrency: int,
    profile: EngineMemoryProfile,
    budget: int,
) -> int:
    def fits(context: int) -> bool:
        kv = arch.kv_bytes(context, profile.kv_bytes_per_element) * concurrency
        return _per_device_need(weights, kv, devices, profile, arch) <= budget

    if not fits(MIN_CONTEXT):
        return 0
    low, high = MIN_CONTEXT, arch.max_context
    if fits(high):
        return high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def _decode_ceiling(
    hw: HardwareReport, pool: DevicePool, arch: ModelArchitecture, weights: int, devices: int
) -> float | None:
    kind = {
        DevicePool.APPLE_GPU: AcceleratorKind.APPLE_GPU,
        DevicePool.NVIDIA_GPU: AcceleratorKind.NVIDIA_GPU,
    }.get(pool)
    if kind is None:
        return None
    bandwidths = [
        acc.mem_bandwidth_gbs for acc in hw.accelerators_of(kind) if acc.mem_bandwidth_gbs
    ]
    if not bandwidths or not weights:
        return None
    active_share = (arch.params_active or arch.params_total) / arch.params_total
    bytes_per_token_per_device = weights * active_share / devices
    return round(min(bandwidths) * 1e9 / bytes_per_token_per_device, 1)


def estimate_fit(
    model: CatalogModel,
    artifact: ModelArtifact,
    engine: EngineId,
    hw: HardwareReport,
    *,
    context_length: int,
    concurrency: int = 1,
    gpu_backends: frozenset[str] | None = None,
) -> FitEstimate:
    """Estimate memory for ``model`` as ``artifact`` on ``engine`` at this context length.

    ``gpu_backends``: what an installed engine reported it can use (see ``device_pool``); a
    llama.cpp build without a GPU backend is planned for the CPU, with a warning.
    """
    arch = model.architecture
    pool = device_pool(engine, hw, gpu_backends)
    profile = memory_profile(engine, pool)
    budget = memory_budget(engine, hw, profile, pool=pool)
    context = min(context_length, arch.max_context)
    weights = artifact.weights_bytes
    kv_total = arch.kv_bytes(context, profile.kv_bytes_per_element) * concurrency

    # Fewest devices that fit comfortably; else fewest that fit at all; else the most devices.
    options = _device_counts(engine, budget, arch)
    comfortable_limit = budget.per_device_bytes * (1 - HEADROOM_TARGET)
    needs = {count: _per_device_need(weights, kv_total, count, profile, arch) for count in options}
    devices = next(
        (count for count in options if needs[count] <= comfortable_limit),
        next((count for count in options if needs[count] <= budget.per_device_bytes), options[-1]),
    )
    per_device = _per_device_need(weights, kv_total, devices, profile, arch)
    fits = per_device <= budget.per_device_bytes
    headroom = 1 - per_device / budget.per_device_bytes if budget.per_device_bytes else -1.0
    overhead = per_device - -(-weights // devices) - _kv_per_device(kv_total, devices, arch)

    notes: list[str] = list(model.notes)
    if context < context_length:
        notes.append(f"{model.display_name} supports at most {arch.max_context:,} tokens.")
    if budget.is_estimate:
        notes.append("The GPU memory limit is macOS's default, so the fit is an estimate.")
    if arch.is_moe and arch.params_active:
        notes.append(
            f"Mixture of experts: about {arch.params_active / 1e9:.1f}B of "
            f"{arch.params_total / 1e9:.1f}B parameters are used per token."
        )
    if devices > 1:
        notes.append(f"Split across {devices} GPUs.")

    warnings: list[str] = []
    unused_gpu = unused_gpu_warning(engine, hw, gpu_backends)
    if unused_gpu is not None:
        warnings.append(unused_gpu)
    available = budget.available_now_bytes
    if fits and available is not None:
        if pool == DevicePool.NVIDIA_GPU and profile.preallocates:
            if available < budget.per_device_bytes:
                warnings.append(
                    f"Other processes use GPU memory: only {_gib(available)} is free, but "
                    f"{engine.display_name} reserves {_gib(budget.per_device_bytes)} per GPU at "
                    "start. Stop them (see `nvidia-smi`) first."
                )
        elif per_device > available:
            warnings.append(
                f"Only {_gib(available)} of memory is free right now; close other apps or "
                "expect slowdowns while the system swaps."
            )

    estimate = FitEstimate(
        model_id=model.id,
        engine=engine,
        artifact=artifact,
        context_length=context,
        concurrency=concurrency,
        devices_used=devices,
        weights_bytes=weights,
        kv_cache_bytes=kv_total,
        overhead_bytes=overhead * devices,
        required_bytes=per_device * devices,
        required_per_device_bytes=per_device,
        budget=budget,
        fits=fits,
        headroom_ratio=round(headroom, 4),
        max_context_that_fits=_max_context(
            arch, weights, devices, concurrency, profile, budget.per_device_bytes
        ),
        est_decode_tokens_per_s=_decode_ceiling(hw, pool, arch, weights, devices),
        notes=tuple(notes),
        warnings=tuple(warnings),
    )
    logger.debug(
        "Fit %s %s on %s: need %s/device of %s (fits=%s, headroom=%.0f%%, devices=%d)",
        model.id,
        artifact.quantization,
        engine.value,
        _gib(per_device),
        _gib(budget.per_device_bytes),
        fits,
        headroom * 100,
        devices,
    )
    return estimate
