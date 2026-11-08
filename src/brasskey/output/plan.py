# SPDX-License-Identifier: Apache-2.0
"""Human view of a run plan (``brasskey run --dry-run``).

Downloads use decimal units (GB, MB/s) as Hugging Face and network speeds do; memory uses GiB as
the hardware does. Every estimate says it is one.
"""

from pathlib import Path

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from brasskey.models.download import DownloadEstimate
from brasskey.models.fit import FitEstimate
from brasskey.output.hardware import LABEL_STYLE, NOTE_STYLE, WARNING_STYLE, format_gib
from brasskey.planner import RunPlan


def format_gb(size_bytes: int) -> str:
    if size_bytes < 10**9:
        return f"{size_bytes / 10**6:.0f} MB"
    return f"{size_bytes / 10**9:.1f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"about {hours} h {rest} min" if rest else f"about {hours} h"


def _home(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home) else path


def _lines(*lines: str | Text | None) -> Text:
    return Text("\n").join(line if isinstance(line, Text) else Text(line) for line in lines if line)


def _memory_text(fit: FitEstimate) -> Text:
    per = " per GPU" if fit.devices_used > 1 else ""
    about = "about " if fit.budget.is_estimate else ""
    headline = (
        f"{about}{format_gib(fit.required_per_device_bytes)}{per} of "
        f"{format_gib(fit.budget.per_device_bytes)} ({fit.headroom_ratio:.0%} to spare), estimate"
    )
    breakdown = (
        f"weights {format_gib(fit.weights_bytes)} + KV cache {format_gib(fit.kv_cache_bytes)} "
        f"for {fit.context_length:,} tokens + overhead {format_gib(fit.overhead_bytes)}"
    )
    if fit.devices_used > 1:
        breakdown += f", split across {fit.devices_used} GPUs"
    max_context = f"up to {fit.max_context_that_fits:,} tokens of context would fit"
    return _lines(headline, breakdown, Text(fit.budget.explanation, style=NOTE_STYLE), max_context)


def _download_text(download: DownloadEstimate | None) -> Text | None:
    if download is None:
        return None
    if download.to_download_bytes == 0:
        return Text(f"already downloaded ({format_gb(download.total_bytes)} in the cache)")
    size = format_gb(download.to_download_bytes)
    if download.cached_bytes >= download.total_bytes / 100:  # ignore a few cached config files
        size = f"{size} of {format_gb(download.total_bytes)} still"
    if download.bandwidth_source == "measured" and download.bandwidth_bytes_per_s:
        speed = download.bandwidth_bytes_per_s / 10**6
        timing = f", {format_duration(download.eta_seconds or 0)} at {speed:.0f} MB/s (measured)"
    elif download.bandwidth_source == "offline":
        timing = " (speed not measured: offline)"
    else:
        timing = " (couldn't measure the download speed)"
    disk = (
        f"{format_gb(download.disk_free_bytes)} free in {_home(download.cache_dir)}"
        if download.disk_free_bytes is not None
        else None
    )
    return _lines(f"{size} to download{timing}", Text(disk, style=NOTE_STYLE) if disk else None)


def _plan_warnings(plan: RunPlan) -> list[str]:
    warnings = list(plan.fit.warnings)
    download = plan.download
    if download is not None and download.fits_on_disk is False:
        warnings.append(
            f"Not enough disk space: {format_gb(download.to_download_bytes)} to download but "
            f"only {format_gb(download.disk_free_bytes or 0)} free in "
            f"{_home(download.cache_dir)}. Free some space or set HF_HOME to a bigger disk."
        )
    return warnings


def _engine_text(plan: RunPlan) -> str:
    install = plan.install
    if install is None:
        return plan.engine.display_name
    backends = ", ".join(backend.capitalize() for backend in install.backends) or "CPU"
    return f"{plan.engine.display_name} {install.version}, {backends} (installed)"


def plan_view(
    plan: RunPlan,
    *,
    engine_line: str | None = None,
    access_line: str | None = None,
    extra_warnings: list[str] | None = None,
) -> RenderableType:
    fit = plan.fit
    model = plan.model
    name = f"{model.display_name} ({model.id}), license {model.license}"
    if not model.verified:
        name += "\nnot in the Brasskey catalog: estimates are unverified"
    bits = model.bits_per_weight(fit.artifact)
    files = (
        f"{fit.artifact.repo_id} @ {fit.artifact.short_revision}, {fit.artifact.quantization} "
        f"({bits:.1f} bits per weight)"
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style=LABEL_STYLE, no_wrap=True)
    table.add_column()
    table.add_row(Text("Model"), Text(name))
    table.add_row(Text("Engine"), Text(engine_line or _engine_text(plan)))
    table.add_row(Text("Files"), Text(files))
    if access_line:
        table.add_row(Text("Access"), Text(access_line))
    table.add_row(Text("Memory"), _memory_text(fit))
    if fit.est_decode_tokens_per_s:
        table.add_row(
            Text("Speed"),
            Text(
                f"at most ~{fit.est_decode_tokens_per_s:.0f} tokens/s when generating "
                "(memory-bandwidth ceiling; real speed is lower)"
            ),
        )
    if (download := _download_text(plan.download)) is not None:
        table.add_row(Text("Download"), download)

    parts: list[RenderableType] = [
        Text("Plan (dry run: nothing is downloaded or started)", style=LABEL_STYLE),
        table,
        Text(""),
        Text("Why", style=LABEL_STYLE),
        *(Text(f"- {reason}") for reason in plan.reasons),
    ]
    if plan.bigger_options:
        parts += [Text(""), Text("Bigger models that also fit", style=LABEL_STYLE)]
        for option in plan.bigger_options:
            memory = format_gib(option.required_per_device_bytes)
            parts.append(
                Text(
                    f"{option.model_id}: {option.engine.display_name} "
                    f"{option.artifact.quantization}, {memory} "
                    f"({option.headroom_ratio:.0%} to spare), "
                    f"{format_gb(option.artifact.download_bytes)} download"
                )
            )
    warnings = _plan_warnings(plan) + list(extra_warnings or [])
    if warnings or fit.notes:
        parts += [Text(""), Text("Things to know", style=LABEL_STYLE)]
        parts += [Text.assemble(("! ", WARNING_STYLE), warning) for warning in warnings]
        parts += [Text.assemble(("i ", LABEL_STYLE), note) for note in fit.notes]
    return Group(*parts)
