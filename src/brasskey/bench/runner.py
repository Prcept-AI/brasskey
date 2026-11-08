# SPDX-License-Identifier: Apache-2.0
"""The order of benchmark runs, their time estimate, and running them.

- One discarded warm-up run per endpoint first (caches, compilation, CUDA graphs).
- Then the measured repeats. Run i uses seed + i, so prompts differ between repeats and a
  prefix cache can't make later runs look faster; both endpoints of a comparison get the same
  seeds.
- When comparing, endpoints alternate and the order flips every repeat (A B, B A, A B ...), so
  slow drifts (heat, background load) affect both alike instead of favoring one.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from brasskey.bench.models import RunMetrics, Workload

logger = logging.getLogger(__name__)

RUN_OVERHEAD_S = 12.0  # GuideLLM start-up, tokenizer loading, finishing in-flight requests
WARMUP_SEED_OFFSET = 1000


@dataclass(frozen=True)
class Step:
    role: str  # "candidate" or "baseline"
    seed: int
    duration_s: int
    measured: bool
    repeat: int  # 0 for the warm-up, then 1..repeats


def schedule(workload: Workload, roles: Sequence[str]) -> list[Step]:
    steps: list[Step] = []
    if workload.warmup_s:
        warm_seed = workload.seed + WARMUP_SEED_OFFSET
        steps += [Step(role, warm_seed, workload.warmup_s, False, 0) for role in roles]
    for index in range(workload.repeats):
        order = list(roles) if index % 2 == 0 else list(reversed(roles))
        steps += [
            Step(role, workload.seed + index, workload.duration_s, True, index + 1)
            for role in order
        ]
    return steps


def estimate_seconds(steps: Sequence[Step]) -> float:
    return sum(step.duration_s + RUN_OVERHEAD_S for step in steps)


def execute(
    steps: Sequence[Step],
    run: Callable[[Step], RunMetrics],
    on_step: Callable[[int, Step], None] | None = None,
) -> dict[str, list[RunMetrics]]:
    """Run every step in order; measured results by role (warm-ups are discarded)."""
    results: dict[str, list[RunMetrics]] = {}
    for number, step in enumerate(steps, start=1):
        if on_step is not None:
            on_step(number, step)
        metrics = run(step)
        logger.debug(
            "%s %s seed=%d: %.1f output tok/s, %d ok, %d errors",
            step.role,
            f"repeat {step.repeat}" if step.measured else "warm-up",
            step.seed,
            metrics.output_tokens_per_s,
            metrics.successful,
            metrics.errored,
        )
        if step.measured:
            results.setdefault(step.role, []).append(metrics)
    return results
