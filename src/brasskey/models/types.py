# SPDX-License-Identifier: Apache-2.0
"""Model catalog types: architecture, downloadable artifacts and catalog entries."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from brasskey.engines.capabilities import ENGINE_FORMATS
from brasskey.engines.types import EngineId, ModelFormat

WEIGHT_SUFFIXES = (".gguf", ".safetensors")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelArchitecture(_Frozen):
    """What decides memory use: parameter counts and the shape of the KV cache."""

    params_total: int = Field(gt=0)
    params_active: int | None = Field(
        default=None, gt=0, description="Mixture-of-experts: parameters used per token."
    )
    num_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    max_context: int = Field(gt=0)
    sliding_window: int | None = Field(default=None, gt=0)
    sliding_layers: int = Field(
        default=0, ge=0, description="Layers with sliding-window attention."
    )
    kv_lora_rank: int | None = Field(default=None, gt=0, description="MLA latent size (DeepSeek).")
    qk_rope_head_dim: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> "ModelArchitecture":
        if self.sliding_layers > self.num_layers:
            raise ValueError("sliding_layers cannot exceed num_layers")
        if self.sliding_layers and self.sliding_window is None:
            raise ValueError("sliding_layers needs sliding_window")
        return self

    @property
    def is_moe(self) -> bool:
        return self.params_active is not None and self.params_active < self.params_total

    @property
    def uses_mla(self) -> bool:
        return self.kv_lora_rank is not None

    def kv_bytes_per_token_per_layer(self, bytes_per_element: float = 2.0) -> float:
        if self.kv_lora_rank is not None:
            # MLA caches one compressed latent plus the rotary key part per token and layer.
            return (self.kv_lora_rank + (self.qk_rope_head_dim or 0)) * bytes_per_element
        return 2 * self.num_kv_heads * self.head_dim * bytes_per_element  # K and V

    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> int:
        """KV cache per token with every layer attending to the full context.

        Llama-3-70B: 2 x 80 layers x 8 KV heads x 128 x 2 bytes = 327,680 bytes (~0.3 MB).
        """
        return round(self.kv_bytes_per_token_per_layer(bytes_per_element) * self.num_layers)

    def kv_bytes(self, context_length: int, bytes_per_element: float = 2.0) -> int:
        """KV cache for one sequence of ``context_length`` tokens (sliding layers are capped)."""
        full_layers = self.num_layers - self.sliding_layers
        window = min(context_length, self.sliding_window or context_length)
        tokens = full_layers * context_length + self.sliding_layers * window
        return round(self.kv_bytes_per_token_per_layer(bytes_per_element) * tokens)


class ArtifactFile(_Frozen):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, description="For large (LFS/Xet) files.")

    @field_validator("sha256")
    @classmethod
    def _hex(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("sha256 must be 64 lower-case hex characters")
        return value


class ModelArtifact(_Frozen):
    """One downloadable form of a model: a repo at a pinned revision, in one format."""

    format: ModelFormat
    repo_id: str
    revision: str = Field(description="Pinned commit SHA.")
    quantization: str = Field(description="e.g. Q4_K_M, Q8_0, 4bit, bf16, fp8, awq-int4")
    files: tuple[ArtifactFile, ...] = Field(min_length=1)

    @property
    def engines(self) -> tuple[EngineId, ...]:
        return tuple(engine for engine, fmt in ENGINE_FORMATS.items() if fmt == self.format)

    @property
    def download_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    @property
    def weights_bytes(self) -> int:
        """Bytes of weight files: about what the engine holds in memory for the weights."""
        return sum(file.size_bytes for file in self.files if file.path.endswith(WEIGHT_SUFFIXES))

    @property
    def short_revision(self) -> str:
        return self.revision[:10]


class CatalogModel(_Frozen):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    display_name: str
    family: str
    license: str
    source_repo: str = Field(description="Where the architecture was read from.")
    quick_start_rank: int | None = Field(
        default=None, description="Higher is more capable; None excludes it from automatic picks."
    )
    verified: bool = Field(
        default=True, description="False for hf: models outside the Brasskey catalog."
    )
    architecture: ModelArchitecture
    artifacts: tuple[ModelArtifact, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()

    def artifacts_for(self, engine: EngineId) -> tuple[ModelArtifact, ...]:
        return tuple(artifact for artifact in self.artifacts if engine in artifact.engines)

    def bits_per_weight(self, artifact: ModelArtifact) -> float:
        return artifact.weights_bytes * 8 / self.architecture.params_total


class Catalog(_Frozen):
    catalog_version: Literal[1] = 1
    generated_at: date
    models: tuple[CatalogModel, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> "Catalog":
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("catalog model ids must be unique")
        return self
