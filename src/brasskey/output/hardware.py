# SPDX-License-Identifier: Apache-2.0
"""Human view of a :class:`HardwareReport` (used by ``brasskey doctor``).

Memory sizes are shown in GiB (binary, as the hardware reports them); bandwidth in GB/s (decimal,
as vendors publish it). Estimates are always labeled as estimates.
"""

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from brasskey.hardware.types import (
    GIB,
    Accelerator,
    AcceleratorKind,
    CpuInfo,
    HardwareReport,
    MemoryInfo,
    ProbeIssue,
)

LABEL_STYLE = "bold"
NOTE_STYLE = "dim"
WARNING_STYLE = "bold yellow"


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / GIB:.1f} GiB"


def _lines(*lines: str | Text | None) -> Text:
    """Join non-empty lines into one Text (cells of the hardware table)."""
    present = [line if isinstance(line, Text) else Text(line) for line in lines if line]
    return Text("\n").join(present)


def _cpu_text(cpu: CpuInfo) -> Text:
    cores = f"{cpu.physical_cores} cores"
    if cpu.performance_cores is not None and cpu.efficiency_cores is not None:
        cores += f" ({cpu.performance_cores} performance, {cpu.efficiency_cores} efficiency)"
    elif cpu.logical_cores != cpu.physical_cores:
        cores += f" ({cpu.logical_cores} threads)"
    extras: list[str] = []
    if cpu.cpu_limit_cores is not None:
        extras.append(f"this process may use {cpu.cpu_limit_cores:g} cores")
    if cpu.rosetta:
        extras.append("Python is running under Rosetta 2")
    features = f"features: {', '.join(cpu.features)}" if cpu.features else None
    return _lines(
        f"{cpu.model_name}, {cores}, {cpu.arch}",
        features,
        Text("; ".join(extras), style=NOTE_STYLE) if extras else None,
    )


def _memory_text(memory: MemoryInfo) -> Text:
    line = f"{format_gib(memory.total_bytes)} total, {format_gib(memory.available_bytes)} available"
    if memory.cgroup_limit_bytes is not None:
        line += f", container limit {format_gib(memory.cgroup_limit_bytes)}"
    return Text(line)


def _accelerator_text(acc: Accelerator) -> Text:
    title = acc.name
    if acc.gpu_cores:
        title += f", {acc.gpu_cores} GPU cores"
    if acc.kind == AcceleratorKind.APPLE_GPU or acc.unified_memory:
        prefix = "about " if acc.memory_is_estimate else ""
        suffix = " (estimate)" if acc.memory_is_estimate else ""
        memory = f"{prefix}{format_gib(acc.memory_total_bytes)} of unified memory usable by the GPU"
        memory += suffix
    else:
        memory = format_gib(acc.memory_total_bytes)
        if acc.memory_free_bytes is not None:
            memory += f" ({format_gib(acc.memory_free_bytes)} free)"
    details: list[str] = []
    if acc.compute_capability:
        details.append(f"compute {acc.compute_capability[0]}.{acc.compute_capability[1]}")
    if acc.mem_bandwidth_gbs:
        details.append(f"{acc.mem_bandwidth_gbs:g} GB/s peak bandwidth")
    if acc.driver_version:
        details.append(f"driver {acc.driver_version}")
    if acc.cuda_driver_version:
        details.append(f"CUDA {acc.cuda_driver_version}")
    if acc.mig_enabled:
        details.append("MIG enabled")
    notes = [Text(note, style=NOTE_STYLE) for note in acc.notes]
    return _lines(title, memory, ", ".join(details) if details else None, *notes)


def _issue_text(issue: ProbeIssue) -> Text:
    marker = ("! ", WARNING_STYLE) if issue.severity == "warning" else ("i ", LABEL_STYLE)
    text = Text.assemble(marker, issue.message)
    if issue.fix:
        text.append("\n  ")
        text.append("Fix: ", style=LABEL_STYLE)
        text.append(issue.fix)
    return text


def hardware_view(report: HardwareReport) -> RenderableType:
    """A compact, plain-language summary of the machine."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=LABEL_STYLE, no_wrap=True)
    table.add_column()
    table.add_row(Text("OS"), Text(report.os.description))
    table.add_row(Text("CPU"), _cpu_text(report.cpu))
    table.add_row(Text("Memory"), _memory_text(report.memory))
    if report.accelerators:
        for acc in report.accelerators:
            table.add_row(Text(f"GPU {acc.index}"), _accelerator_text(acc))
    else:
        table.add_row(Text("GPU"), Text("none found; models will run on the CPU"))

    parts: list[RenderableType] = [Text("This machine", style=LABEL_STYLE), table]
    if report.issues:
        parts += [Text(""), Text("Things to know", style=LABEL_STYLE)]
        parts += [_issue_text(issue) for issue in report.issues]
    parts += [Text(""), Text.assemble(("Fingerprint: ", NOTE_STYLE), report.fingerprint())]
    return Group(*parts)
