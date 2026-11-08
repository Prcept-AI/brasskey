# SPDX-License-Identifier: Apache-2.0
"""Statistics for repeated benchmark runs: intervals, noise, and comparison verdicts.

Pure functions, no third-party dependencies (CONTRIBUTING.md: measure, don't guess; and a report
never claims a gain the data can't support):

- each metric is summarized over the repeated runs with a 95% Student-t interval for its mean
  and a coefficient of variation (CV) that flags noisy measurements;
- two endpoints are compared on the log scale (ratios, so "12% faster" means the same at any
  speed) with Welch's t interval, which doesn't assume equal variances;
- the verdict follows ADR 0011: with the improvement interval [lo, hi] (positive means
  better) and the target t (+margin for "better", -margin for "not worse"), PASS if lo >= t,
  FAIL if hi < t, otherwise INCONCLUSIVE.

Student-t quantiles come from the regularized incomplete beta function, evaluated with a
continued fraction (modified Lentz method) and inverted by bisection.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist
from typing import Literal

CONFIDENCE = 0.95
NOISY_CV = 0.10  # runs that differ by more than 10% (CV) are flagged as noisy
_EPS = 1e-15
_TINY = 1e-300
LARGE_DF = 1e7

Better = Literal["higher", "lower"]
Goal = Literal["better", "not-worse"]


class Verdict(StrEnum):
    PASS = "PASS"  # noqa: S105 (a verdict, not a password)
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


def _continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (DLMF 8.17.22), modified Lentz."""
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    result = d
    for m in range(1, 500):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > _TINY else _TINY)
            c = 1.0 + numerator / c
            c = c if abs(c) > _TINY else _TINY
            result *= c * d
        if abs(c * d - 1.0) < _EPS:
            break
    return result


def regularized_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) for 0 <= x <= 1 and a, b > 0."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    tail = 0.5 * regularized_beta(df / (df + t * t), df / 2.0, 0.5)
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """The value t with P(T <= t) = p for Student's t with ``df`` degrees of freedom."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    if df <= 0:
        raise ValueError("df must be positive")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    if p == 0.5:
        return 0.0
    if df > LARGE_DF:  # the t distribution is the normal one to well within 1e-6 here
        return NormalDist().inv_cdf(p)
    high = 2.0
    while student_t_cdf(high, df) < p:
        high *= 2.0
    low = 0.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, high):
            break
    return (low + high) / 2.0


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    """Sample variance (n - 1)."""
    center = _mean(values)
    return math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float | None  # None with fewer than two runs
    high: float | None
    cv: float | None  # standard deviation / mean


def mean_interval(values: Sequence[float], confidence: float = CONFIDENCE) -> Interval:
    """Mean with a two-sided Student-t interval and the coefficient of variation."""
    if not values:
        raise ValueError("no values")
    center = _mean(values)
    if len(values) < 2:
        return Interval(center, None, None, None)
    spread = math.sqrt(_variance(values))
    half = t_quantile(0.5 + confidence / 2.0, len(values) - 1) * spread / math.sqrt(len(values))
    cv = spread / center if center > 0 else None
    return Interval(center, center - half, center + half, cv)


@dataclass(frozen=True)
class LogComparison:
    """Candidate vs baseline on the log scale.

    ``change`` is the candidate/baseline ratio of geometric means minus one (e.g. +0.12 for 12%
    more), with its interval. ``improvement`` is the log ratio oriented so that positive means
    better (it decides the verdict), with ``target`` from the goal and margin.
    """

    change: float
    change_low: float
    change_high: float
    improvement: float
    improvement_low: float
    improvement_high: float
    target: float
    df: float
    verdict: Verdict
    reason: str


def compare_runs(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    better: Better,
    goal: Goal = "better",
    margin: float = 0.0,
    confidence: float = CONFIDENCE,
) -> LogComparison | str:
    """Welch's t interval on log values, and the verdict. Returns a reason when it can't."""
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be at least 0 and below 1")
    if len(baseline) < 2 or len(candidate) < 2:
        return "At least two runs per endpoint are needed for an interval."
    if any(value <= 0 for value in (*baseline, *candidate)):
        return "A run measured zero, so ratios can't be computed (see the errors above)."
    if min(baseline) == max(baseline) and min(candidate) == max(candidate):
        # Every run gave the same value (e.g. a request count over a short run): the precision
        # is limited by that granularity, which an interval of zero width would hide.
        return (
            "Every run gave the same value, so the measurement is too coarse for an interval; "
            "longer runs (--duration) fix this."
        )
    log_base = [math.log(value) for value in baseline]
    log_cand = [math.log(value) for value in candidate]
    raw = _mean(log_cand) - _mean(log_base)
    var_base = _variance(log_base) / len(log_base)
    var_cand = _variance(log_cand) / len(log_cand)
    se = math.sqrt(var_base + var_cand)
    denominator = var_base**2 / (len(log_base) - 1) + var_cand**2 / (len(log_cand) - 1)
    df = (var_base + var_cand) ** 2 / denominator if denominator > 0 else math.inf
    half = t_quantile(0.5 + confidence / 2.0, df) * se if se > 0 else 0.0
    sign = 1.0 if better == "higher" else -1.0
    improvement = sign * raw
    low, high = improvement - half, improvement + half
    target = math.log1p(margin) if goal == "better" else math.log1p(-margin)
    if low >= target:
        verdict = Verdict.PASS
    elif high < target:
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.INCONCLUSIVE
    reason = _reason(verdict, goal, margin, confidence)
    return LogComparison(
        change=math.expm1(raw),
        change_low=math.expm1(raw - half),
        change_high=math.expm1(raw + half),
        improvement=improvement,
        improvement_low=low,
        improvement_high=high,
        target=target,
        df=df,
        verdict=verdict,
        reason=reason,
    )


def _reason(verdict: Verdict, goal: Goal, margin: float, confidence: float) -> str:
    level = f"{confidence:.0%} confidence"
    if goal == "better":
        claim = f"better by at least {margin:.0%}" if margin else "better"
    else:
        claim = f"no more than {margin:.0%} worse" if margin else "no worse"
    if verdict == Verdict.PASS:
        return f"The candidate is {claim} ({level})."
    if verdict == Verdict.FAIL:
        return f"The candidate is not {claim} ({level})."
    return (
        f"The runs can't tell whether the candidate is {claim} ({level}); more repeats or a "
        "quieter machine narrow the interval."
    )
