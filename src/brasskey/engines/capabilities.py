# SPDX-License-Identifier: Apache-2.0
"""What each engine can run, on which hardware, and how it uses memory.

These are fixed rules for v0.1 (there is no registry of measurements yet), so every result that
depends on them is labeled a rule or an estimate. The engine adapters (milestones d, e, g) reuse
these rules and add what only a real installation can tell: whether the engine is installed, its
version, and which backends it was built with.

Rules for NVIDIA quantization support are conservative and are checked on real GPUs with the
``gpu`` tests (milestone g, docs/gpu-testing.md).
"""

from collections.abc import Sequence

from brasskey.engines.types import (
    DevicePool,
    EngineId,
    EngineMemoryProfile,
    ModelFormat,
    SupportCheck,
)
from brasskey.hardware.types import GIB, Accelerator, AcceleratorKind, HardwareReport

GPU_SERVER_ENGINES = frozenset({EngineId.VLLM, EngineId.SGLANG})  # Linux + NVIDIA, preallocating

ENGINE_FORMATS: dict[EngineId, ModelFormat] = {
    EngineId.LLAMACPP: ModelFormat.GGUF,
    EngineId.MLX: ModelFormat.MLX,
    EngineId.VLLM: ModelFormat.SAFETENSORS,
    EngineId.SGLANG: ModelFormat.SAFETENSORS,
}

# Minimum NVIDIA compute capability per engine, and per quantization where it matters.
# vLLM 0.32 and SGLang 0.5.22 are built with CUDA 13.0, which no longer supports Volta (7.0).
MIN_COMPUTE_CAPABILITY: dict[EngineId, tuple[int, int]] = {
    EngineId.VLLM: (7, 5),
    EngineId.SGLANG: (7, 5),
}

# The oldest NVIDIA driver branch for each CUDA version (NVIDIA's CUDA release notes, table 3).
DRIVER_FOR_CUDA: dict[tuple[int, int], str] = {
    (13, 0): "580",
    (12, 9): "575",
    (12, 8): "570",
    (12, 6): "560",
    (12, 4): "550",
}
MIN_CC_FOR_QUANTIZATION: dict[EngineId, dict[str, tuple[int, int]]] = {
    EngineId.VLLM: {"fp8": (8, 0), "awq-int4": (7, 5)},  # FP8 on Ampere runs weight-only
    EngineId.SGLANG: {"fp8": (8, 9), "awq-int4": (7, 5)},
}
BF16_MIN_CC = (8, 0)

# Memory profiles (estimates; see docs/decisions/0004-model-fit-and-planning.md).
_LLAMACPP = EngineMemoryProfile(max_fraction=0.95, overhead_bytes=GIB // 2, overhead_fraction=0.02)
_LLAMACPP_CPU = EngineMemoryProfile(
    max_fraction=1.0, overhead_bytes=GIB // 2, overhead_fraction=0.02
)
_LLAMACPP_APPLE = EngineMemoryProfile(
    max_fraction=1.0, overhead_bytes=GIB // 2, overhead_fraction=0.02
)
_MLX = EngineMemoryProfile(max_fraction=1.0, overhead_bytes=3 * GIB // 4, overhead_fraction=0.05)
_VLLM = EngineMemoryProfile(
    max_fraction=0.90, overhead_bytes=2 * GIB, overhead_fraction=0.05, preallocates=True
)
_SGLANG = EngineMemoryProfile(
    max_fraction=0.85, overhead_bytes=2 * GIB, overhead_fraction=0.05, preallocates=True
)


def _cc_text(cc: tuple[int, int]) -> str:
    return f"{cc[0]}.{cc[1]}"


def engine_preference(hw: HardwareReport) -> list[EngineId]:
    """Engines to try, best first, for this hardware (a fixed rule until measurements exist).

    SGLang is used when asked for (`--engine sglang`); vLLM is the safer general default.
    """
    if hw.accelerators_of(AcceleratorKind.APPLE_GPU):
        return [EngineId.MLX, EngineId.LLAMACPP]
    if hw.accelerators_of(AcceleratorKind.NVIDIA_GPU) and hw.os.family == "linux":
        return [EngineId.VLLM, EngineId.LLAMACPP]
    return [EngineId.LLAMACPP]


def device_pool(engine: EngineId, hw: HardwareReport) -> DevicePool:
    """Where this engine would place the model on this machine."""
    if engine == EngineId.MLX:
        return DevicePool.APPLE_GPU
    if engine in (EngineId.VLLM, EngineId.SGLANG):
        return DevicePool.NVIDIA_GPU
    if hw.accelerators_of(AcceleratorKind.APPLE_GPU):
        return DevicePool.APPLE_GPU  # llama.cpp with Metal
    if hw.accelerators_of(AcceleratorKind.NVIDIA_GPU):
        return DevicePool.NVIDIA_GPU  # llama.cpp with CUDA
    return DevicePool.CPU


def memory_profile(engine: EngineId, pool: DevicePool) -> EngineMemoryProfile:
    if engine == EngineId.MLX:
        return _MLX
    if engine == EngineId.VLLM:
        return _VLLM
    if engine == EngineId.SGLANG:
        return _SGLANG
    if pool == DevicePool.CPU:
        return _LLAMACPP_CPU
    if pool == DevicePool.APPLE_GPU:
        return _LLAMACPP_APPLE
    return _LLAMACPP


def check_engine(engine: EngineId, hw: HardwareReport) -> SupportCheck:
    """Can this engine run on this machine at all?"""
    name = engine.display_name
    if engine == EngineId.LLAMACPP:
        return SupportCheck(supported=True)
    if engine == EngineId.MLX:
        if hw.os.family != "darwin" or not hw.accelerators_of(AcceleratorKind.APPLE_GPU):
            return SupportCheck(supported=False, reasons=("MLX runs only on Apple Silicon Macs.",))
        return SupportCheck(supported=True)

    # vLLM and SGLang: Linux with an NVIDIA GPU.
    if hw.os.family != "linux":
        return SupportCheck(
            supported=False, reasons=(f"{name} runs on Linux (or WSL2) with an NVIDIA GPU.",)
        )
    gpus = hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
    if not gpus:
        return SupportCheck(supported=False, reasons=(f"{name} needs an NVIDIA GPU; none found.",))
    mig = [gpu.index for gpu in gpus if gpu.mig_enabled]
    if mig:
        listed = ", ".join(str(index) for index in mig)
        return SupportCheck(
            supported=False,
            reasons=(
                f"MIG is enabled on NVIDIA GPU {listed}, and Brasskey v0.1 plans {name} for whole "
                "GPUs. Disable MIG, or hide those GPUs with CUDA_VISIBLE_DEVICES.",
            ),
        )
    minimum = MIN_COMPUTE_CAPABILITY[engine]
    capable = [gpu for gpu in gpus if (gpu.compute_capability or (0, 0)) >= minimum]
    if not capable:
        return SupportCheck(
            supported=False,
            reasons=(
                f"{name} needs an NVIDIA GPU with compute capability {_cc_text(minimum)} or "
                f"newer; this machine has {_cc_text(gpus[0].compute_capability or (0, 0))}.",
            ),
        )
    return SupportCheck(supported=True)


def check_quantization(engine: EngineId, quantization: str, hw: HardwareReport) -> SupportCheck:
    """Can this engine run weights in this quantization on these GPUs?"""
    if engine not in (EngineId.VLLM, EngineId.SGLANG):
        return SupportCheck(supported=True)
    gpus = hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
    lowest = min((gpu.compute_capability or (0, 0) for gpu in gpus), default=(0, 0))
    needed = MIN_CC_FOR_QUANTIZATION[engine].get(quantization)
    if needed is not None and lowest < needed:
        return SupportCheck(
            supported=False,
            reasons=(
                f"{engine.display_name} needs compute capability {_cc_text(needed)}+ for "
                f"{quantization} weights; this GPU has {_cc_text(lowest)}.",
            ),
        )
    if quantization == "bf16" and lowest < BF16_MIN_CC:
        return SupportCheck(
            supported=True,
            notes=("This GPU has no bfloat16 support, so the engine runs the weights as float16.",),
        )
    return SupportCheck(supported=True)


def parse_cuda_version(text: str | None) -> tuple[int, int] | None:
    """``"13.0"`` or ``"12.9.1"`` -> (13, 0) / (12, 9); None when unreadable."""
    if not text:
        return None
    parts = text.strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def check_cuda_driver(
    engine: EngineId, cuda_version: str | None, hw: HardwareReport
) -> SupportCheck:
    """Is the NVIDIA driver new enough for the CUDA version the engine was built with?

    A mismatch is a warning, not a refusal: datacenter GPUs can run newer CUDA on older drivers
    through NVIDIA's forward-compatibility libraries (which the engine images include), and the
    engine's own start-up error is explained if it really can't.
    """
    needed = parse_cuda_version(cuda_version)
    gpus = hw.accelerators_of(AcceleratorKind.NVIDIA_GPU)
    if engine not in GPU_SERVER_ENGINES or needed is None or not gpus:
        return SupportCheck(supported=True)
    supported = parse_cuda_version(gpus[0].cuda_driver_version)
    if supported is None or supported >= needed:
        return SupportCheck(supported=True)
    driver = DRIVER_FOR_CUDA.get(needed)
    newer = f"driver {driver} or newer" if driver else "a newer driver"
    return SupportCheck(
        supported=True,
        notes=(
            f"The NVIDIA driver ({gpus[0].driver_version or 'unknown version'}) supports CUDA up "
            f"to {gpus[0].cuda_driver_version}, but {engine.display_name} is built for CUDA "
            f"{cuda_version}. It starts only on datacenter GPUs (forward compatibility); if it "
            f"fails, install {newer}.",
        ),
    )


def choose_gpus(gpus: Sequence[Accelerator], count: int) -> tuple[int, ...]:
    """NVML indices of the ``count`` GPUs with the most free memory (ties: lowest index).

    The fit assumed the least-free GPU of the machine, so any choice here is at least as roomy.
    """
    ranked = sorted(gpus, key=lambda gpu: (-(gpu.memory_free_bytes or 0), gpu.index))
    return tuple(sorted(gpu.index for gpu in ranked[: max(count, 1)]))
