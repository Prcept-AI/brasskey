# SPDX-License-Identifier: Apache-2.0
"""``brasskey.lock``: the resolved plan, enough to run the same thing again anywhere.

It pins the model files (repo, commit, sizes, sha256), the engine and its version, the settings
that shape the launch, the hardware fingerprint it was resolved on, the memory estimate, and what
was measured when it started. Machine-specific paths and secrets are never written: the model file
appears as ``{model}`` in the recorded arguments, and the API key only by variable name.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brasskey.bench.models import MetricSummary, Workload
from brasskey.engines.types import EngineId
from brasskey.models.types import ModelArchitecture, ModelArtifact

LOCK_FILE = "brasskey.lock"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LockedHardware(_Strict):
    fingerprint: str
    summary: str


class LockedModel(_Strict):
    id: str
    display_name: str
    family: str
    license: str
    source_repo: str
    verified: bool
    architecture: ModelArchitecture


class LockedEngine(_Strict):
    id: EngineId
    version: str | None = Field(description="Version detected when the lock was written.")
    runtime: Literal["process", "uvx", "docker"] | None = Field(
        default=None, description="How the engine ran (added in milestone g)."
    )
    image: str | None = Field(
        default=None, description="Docker runtime: the pinned image, repository:tag@sha256:..."
    )
    args: tuple[str, ...] = Field(
        default=(), description="Launch arguments (informational; {model} = the model file)."
    )


class LockedServe(_Strict):
    host: str
    port: int | None
    context_length: int
    concurrency: int
    api_key_env: str | None = None
    api_key_source: str | None = Field(
        default=None,
        description='Where the key came from ("env:NAME" or "keyring"), never the key.',
    )
    tls: bool = False


class LockedEstimate(_Strict):
    required_per_device_bytes: int
    budget_per_device_bytes: int
    devices_used: int
    headroom_ratio: float
    is_estimate: Literal[True] = True


class LockedMeasurement(_Strict):
    started: bool = Field(description="True once the engine reported ready with these settings.")
    startup_seconds: float | None = None
    measured_at: datetime | None = None


class LockedBenchmark(_Strict):
    """A `brasskey bench --lock` result: numbers only, no prompts or responses."""

    measured_at: datetime
    tool: str = Field(description='e.g. "guidellm 0.8.1"')
    model: str = Field(description="The model name the endpoint served.")
    workload: Workload
    metrics: tuple[MetricSummary, ...]


class Lock(_Strict):
    lock_version: Literal[1] = 1
    generated_by: str
    generated_at: datetime
    intent_sha256: str | None = Field(
        default=None, description="sha256 of the brasskey.yaml it came from, if any."
    )
    hardware: LockedHardware
    model: LockedModel
    artifact: ModelArtifact
    engine: LockedEngine
    serve: LockedServe
    estimate: LockedEstimate
    measured: LockedMeasurement
    benchmarks: tuple[LockedBenchmark, ...] = Field(
        default=(), description="Appended by `brasskey bench --lock` (added in milestone h)."
    )
