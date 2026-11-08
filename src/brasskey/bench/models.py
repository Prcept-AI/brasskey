# SPDX-License-Identifier: Apache-2.0
"""What a benchmark measures and saves. Numbers only: never prompts or responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brasskey.bench.stats import Better, Goal, Verdict

REPORT_FORMAT = "brasskey.bench/v1"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Workload(_Frozen):
    """The traffic each run sends: synthetic prompts of a fixed shape, at fixed concurrency."""

    prompt_tokens: int = Field(default=256, ge=1)
    output_tokens: int = Field(default=128, ge=1)
    concurrency: int = Field(default=4, ge=1)
    duration_s: int = Field(default=30, ge=5, description="Measured seconds per run.")
    repeats: int = Field(default=3, ge=1, description="Measured runs per endpoint.")
    warmup_s: int = Field(default=10, ge=0, description="One discarded run first (0: none).")
    seed: int = Field(default=42, description="Run i uses seed + i; the warm-up uses seed + 1000.")


class RunMetrics(_Frozen):
    """One GuideLLM run, reduced to the numbers Brasskey reports."""

    seed: int
    duration_s: float
    successful: int
    errored: int
    incomplete: int
    output_tokens_per_s: float
    requests_per_s: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    itl_ms_p50: float
    latency_s_p50: float
    latency_s_p95: float


class MetricInfo(_Frozen):
    key: str  # field of RunMetrics
    option: str  # value of --metric
    label: str
    unit: str
    better: Better


METRICS: tuple[MetricInfo, ...] = (
    MetricInfo(key="output_tokens_per_s", option="output-throughput", label="Output tokens/s",
               unit="tok/s", better="higher"),
    MetricInfo(key="requests_per_s", option="request-throughput", label="Requests/s",
               unit="req/s", better="higher"),
    MetricInfo(key="ttft_ms_p50", option="ttft", label="Time to first token (median)",
               unit="ms", better="lower"),
    MetricInfo(key="itl_ms_p50", option="itl", label="Inter-token latency (median)",
               unit="ms", better="lower"),
    MetricInfo(key="latency_s_p95", option="latency", label="Request latency (p95)",
               unit="s", better="lower"),
)  # fmt: skip
METRIC_OPTIONS = tuple(info.option for info in METRICS)


def metric_by_option(option: str) -> MetricInfo:
    return next(info for info in METRICS if info.option == option)


class MetricSummary(_Frozen):
    key: str
    label: str
    unit: str
    better: Better
    values: tuple[float, ...]
    mean: float
    ci_low: float | None = None
    ci_high: float | None = None
    cv: float | None = None
    noisy: bool = False


class EndpointResult(_Frozen):
    url: str  # API base URL, e.g. http://127.0.0.1:8000/v1 (never with credentials)
    model: str
    engine: str | None = Field(default=None, description="When it is a Brasskey server.")
    engine_version: str | None = None
    runs: tuple[RunMetrics, ...]
    metrics: tuple[MetricSummary, ...]

    def metric(self, key: str) -> MetricSummary:
        return next(summary for summary in self.metrics if summary.key == key)


class MetricComparison(_Frozen):
    key: str
    label: str
    better: Better
    change: float | None = Field(
        default=None, description="Candidate / baseline - 1 (geometric means)."
    )
    change_low: float | None = None
    change_high: float | None = None
    verdict: Verdict
    reason: str


class ComparisonResult(_Frozen):
    baseline_source: Literal["endpoint", "file"]
    goal: Goal
    margin: float
    primary_metric: str
    metrics: tuple[MetricComparison, ...]
    verdict: Verdict
    reason: str


class BenchReport(_Frozen):
    """What `brasskey bench --save` writes, and what `--baseline` reads back."""

    format: Literal["brasskey.bench/v1"] = REPORT_FORMAT
    created_at: datetime
    brasskey_version: str
    tool: str = Field(description='e.g. "guidellm 0.8.1"')
    workload: Workload
    candidate: EndpointResult
    baseline: EndpointResult | None = None
    comparison: ComparisonResult | None = None
