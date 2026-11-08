# SPDX-License-Identifier: Apache-2.0
"""Engine identities and the small value types shared by the planner and the adapters."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EngineId(StrEnum):
    LLAMACPP = "llamacpp"
    MLX = "mlx"
    VLLM = "vllm"
    SGLANG = "sglang"

    @property
    def display_name(self) -> str:
        return ENGINE_DISPLAY_NAMES[self]


ENGINE_DISPLAY_NAMES = {
    EngineId.LLAMACPP: "llama.cpp",
    EngineId.MLX: "MLX",
    EngineId.VLLM: "vLLM",
    EngineId.SGLANG: "SGLang",
}


class ModelFormat(StrEnum):
    GGUF = "gguf"  # llama.cpp
    MLX = "mlx"  # MLX (Apple Silicon)
    SAFETENSORS = "safetensors"  # vLLM, SGLang


class DevicePool(StrEnum):
    """Where an engine would keep the model."""

    APPLE_GPU = "apple_gpu"  # unified memory, up to the GPU working-set limit
    NVIDIA_GPU = "nvidia_gpu"  # VRAM on one or more NVIDIA GPUs
    CPU = "cpu"  # system RAM


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SupportCheck(_Frozen):
    """Whether an engine can run something here, with plain-language reasons when it can't."""

    supported: bool
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class EngineMemoryProfile(_Frozen):
    """How an engine uses device memory. Values are estimates, calibrated by measurement later."""

    max_fraction: float = Field(
        gt=0, le=1, description="Share of each device's memory the engine may use."
    )
    overhead_bytes: int = Field(ge=0, description="Fixed runtime overhead per device.")
    overhead_fraction: float = Field(
        ge=0, description="Extra overhead (activations, buffers) as a share of the weights."
    )
    kv_bytes_per_element: float = Field(default=2.0, gt=0, description="KV cache precision.")
    preallocates: bool = Field(
        default=False, description="Reserves its whole budget at start (vLLM, SGLang)."
    )
