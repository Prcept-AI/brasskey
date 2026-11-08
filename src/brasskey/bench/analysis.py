# SPDX-License-Identifier: Apache-2.0
"""Turn repeated runs into metric summaries, and two endpoints into a verdict (pure functions)."""

import logging
from collections.abc import Sequence

from brasskey.bench.models import (
    METRICS,
    ComparisonResult,
    EndpointResult,
    MetricComparison,
    MetricInfo,
    MetricSummary,
    RunMetrics,
)
from brasskey.bench.stats import NOISY_CV, Goal, LogComparison, Verdict, compare_runs, mean_interval

logger = logging.getLogger(__name__)


def _values(runs: Sequence[RunMetrics], info: MetricInfo) -> tuple[float, ...]:
    return tuple(float(getattr(run, info.key)) for run in runs)


def _steps(runs: Sequence[RunMetrics], info: MetricInfo) -> tuple[float, ...] | None:
    """Each run's relative resolution for rates of finished requests (one request = 1/n).

    Without it, two runs that both finished 13 requests would claim an interval of zero width.
    """
    if not info.counted:
        return None
    return tuple(1.0 / max(run.successful, 1) for run in runs)


def summarize(runs: Sequence[RunMetrics]) -> tuple[MetricSummary, ...]:
    summaries: list[MetricSummary] = []
    for info in METRICS:
        values = _values(runs, info)
        interval = mean_interval(values, rel_steps=_steps(runs, info))
        summaries.append(
            MetricSummary(
                key=info.key,
                label=info.label,
                unit=info.unit,
                better=info.better,
                values=values,
                mean=interval.mean,
                ci_low=interval.low,
                ci_high=interval.high,
                cv=interval.cv,
                noisy=interval.cv is not None and interval.cv > NOISY_CV,
            )
        )
    return tuple(summaries)


def compare(
    baseline: EndpointResult,
    candidate: EndpointResult,
    *,
    goal: Goal,
    margin: float,
    primary: str,
    baseline_source: str,
) -> ComparisonResult:
    """Every metric compared; the primary metric decides the overall verdict."""
    compared: list[MetricComparison] = []
    for info in METRICS:
        result = compare_runs(
            baseline.metric(info.key).values,
            candidate.metric(info.key).values,
            better=info.better,
            goal=goal,
            margin=margin,
            baseline_steps=_steps(baseline.runs, info),
            candidate_steps=_steps(candidate.runs, info),
        )
        if isinstance(result, LogComparison):
            compared.append(
                MetricComparison(
                    key=info.key,
                    label=info.label,
                    better=info.better,
                    change=result.change,
                    change_low=result.change_low,
                    change_high=result.change_high,
                    verdict=result.verdict,
                    reason=result.reason,
                )
            )
        else:  # not enough data for an interval
            compared.append(
                MetricComparison(
                    key=info.key,
                    label=info.label,
                    better=info.better,
                    verdict=Verdict.INCONCLUSIVE,
                    reason=result,
                )
            )
    deciding = next(item for item in compared if item.key == primary)
    logger.debug(
        "Comparison on %s: %s (change %s)", primary, deciding.verdict.value, deciding.change
    )
    return ComparisonResult(
        baseline_source="file" if baseline_source == "file" else "endpoint",
        goal=goal,
        margin=margin,
        primary_metric=primary,
        metrics=tuple(compared),
        verdict=deciding.verdict,
        reason=f"{deciding.label}: {deciding.reason}",
    )
