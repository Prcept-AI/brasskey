# SPDX-License-Identifier: Apache-2.0
"""Human view of ``brasskey bench``: metrics with intervals, and the comparison verdict."""

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from brasskey.bench.models import BenchReport, EndpointResult, MetricSummary, Workload
from brasskey.bench.stats import NOISY_CV, Verdict
from brasskey.output.hardware import LABEL_STYLE, NOTE_STYLE, WARNING_STYLE

VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.INCONCLUSIVE: "bold yellow",
}


def format_value(value: float, unit: str) -> str:
    if unit == "ms":
        text = f"{value:.2f}" if value < 10 else f"{value:.1f}" if value < 1000 else f"{value:.0f}"
    elif unit == "s":
        text = f"{value:.3f}" if value < 1 else f"{value:.2f}"
    else:
        text = f"{value:.2f}" if value < 10 else f"{value:.1f}"
    return f"{text} {unit}"


def goal_text(goal: str, margin: float) -> str:
    if goal == "better":
        return f"better by at least {margin:.0%}" if margin else "better"
    return f"no more than {margin:.0%} worse" if margin else "no worse"


def _percent(value: float) -> str:
    return f"{value:+.1%}"


def workload_text(workload: Workload) -> str:
    warmup = f" after a {workload.warmup_s} s warm-up" if workload.warmup_s else ""
    runs = "run" if workload.repeats == 1 else "runs"
    return (
        f"{workload.prompt_tokens}-token prompts, {workload.output_tokens}-token answers, "
        f"{workload.concurrency} concurrent; {workload.repeats} {runs} x {workload.duration_s} s"
        f"{warmup}"
    )


def endpoint_text(result: EndpointResult) -> str:
    engine = f" on {result.engine} {result.engine_version or ''}".rstrip() if result.engine else ""
    return f"{result.model}{engine} at {result.url}"


def _interval(summary: MetricSummary) -> str:
    if summary.ci_low is None or summary.ci_high is None:
        return "needs 2+ runs"
    low = format_value(max(summary.ci_low, 0.0), summary.unit)
    return f"{low.split()[0]} to {format_value(summary.ci_high, summary.unit)}"


def _metrics_table(result: EndpointResult) -> Table:
    table = Table(box=None, pad_edge=False, show_edge=False, header_style=LABEL_STYLE)
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column("Run-to-run", justify="right")
    for summary in result.metrics:
        spread = f"{summary.cv:.1%}" if summary.cv is not None else "-"
        table.add_row(
            summary.label,
            format_value(summary.mean, summary.unit),
            _interval(summary),
            Text(spread, style=WARNING_STYLE if summary.noisy else ""),
        )
    return table


def _warnings(result: EndpointResult) -> list[Text]:
    lines: list[Text] = []
    errored = sum(run.errored for run in result.runs)
    total = sum(run.errored + run.successful for run in result.runs)
    if errored:
        lines.append(
            Text(
                f"! {errored} of {total} requests to {result.url} failed; the numbers cover the "
                "successful ones.",
                style=WARNING_STYLE,
            )
        )
    noisy = [summary.label for summary in result.metrics if summary.noisy]
    if noisy:
        lines.append(
            Text(
                f"! Noisy: {', '.join(noisy)} varied more than {NOISY_CV:.0%} between runs. Close "
                "other programs, or add --repeats for a narrower interval.",
                style=WARNING_STYLE,
            )
        )
    return lines


def bench_view(report: BenchReport) -> RenderableType:
    lines: list[RenderableType] = []
    comparison = report.comparison
    if report.baseline is None or comparison is None:
        lines.append(Text(f"Benchmark of {endpoint_text(report.candidate)}", style=LABEL_STYLE))
    else:
        lines.append(Text(f"Candidate  {endpoint_text(report.candidate)}", style=LABEL_STYLE))
        source = " (saved results)" if comparison.baseline_source == "file" else ""
        lines.append(
            Text(f"Baseline   {endpoint_text(report.baseline)}{source}", style=LABEL_STYLE)
        )
    lines.append(Text(f"Workload   {workload_text(report.workload)}"))
    lines.append(Text(f"Tool       {report.tool}", style=NOTE_STYLE))
    lines.append(Text(""))

    if report.baseline is None or comparison is None:
        lines.append(_metrics_table(report.candidate))
        lines.extend(_warnings(report.candidate))
        return Group(*lines)

    table = Table(box=None, pad_edge=False, show_edge=False, header_style=LABEL_STYLE)
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Change (95% interval)", justify="right")
    table.add_column("")
    for item in comparison.metrics:
        base = report.baseline.metric(item.key)
        cand = report.candidate.metric(item.key)
        if item.change is not None and item.change_low is not None and item.change_high is not None:
            change = (
                f"{_percent(item.change)} ({_percent(item.change_low)} to "
                f"{_percent(item.change_high)})"
            )
        else:
            change = "-"
        marker = Text(item.verdict.value, style=VERDICT_STYLE[item.verdict])
        if item.key != comparison.primary_metric:
            marker.stylize("dim")
        table.add_row(
            item.label,
            format_value(base.mean, base.unit),
            format_value(cand.mean, cand.unit),
            change,
            marker,
        )
    lines.append(table)
    lines.extend(_warnings(report.baseline))
    lines.extend(_warnings(report.candidate))
    lines.append(Text(""))
    primary = next(m.label for m in comparison.metrics if m.key == comparison.primary_metric)
    lines.append(
        Text.assemble(
            ("Verdict: ", LABEL_STYLE),
            (comparison.verdict.value, VERDICT_STYLE[comparison.verdict]),
            f"  {comparison.reason}",
        )
    )
    lines.append(
        Text(
            f"Goal: the candidate is {goal_text(comparison.goal, comparison.margin)}, decided by "
            f"{primary} (the other rows are for information).",
            style=NOTE_STYLE,
        )
    )
    return Group(*lines)
