# SPDX-License-Identifier: Apache-2.0
"""Choose what ``brasskey run`` would start: engine, model and artifact (the run plan).

Rules (ADR 0002 and ADR 0004):
- Engines: the requested one, or the default order for this hardware (MLX then llama.cpp on
  Apple Silicon; vLLM then llama.cpp with NVIDIA GPUs on Linux; llama.cpp elsewhere).
- Artifact: the best-quality form that fits with at least 25% of the budget free; otherwise the
  best that fits at all. For vLLM/SGLang: bf16, then fp8, then AWQ int4.
- Quick start (no model named): the most capable catalog model that fits comfortably and needs
  at most about 5 GB of downloading (already-cached files don't count); bigger options that fit
  are listed so the user can choose.

This module does no I/O itself: hardware, catalog, cache lookups and Hugging Face access are passed
in, so it is easy to test and reuse from a future API.
"""

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from brasskey.engines.base import EngineInstall
from brasskey.engines.capabilities import (
    ENGINE_FORMATS,
    check_engine,
    check_quantization,
    engine_preference,
)
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.types import GIB, AcceleratorKind, HardwareReport
from brasskey.models.catalog import find_model
from brasskey.models.download import DownloadEstimate
from brasskey.models.fit import HEADROOM_TARGET, MIN_CONTEXT, FitEstimate, estimate_fit
from brasskey.models.hub import HubClient, ModelSourceError, resolve_hf_model
from brasskey.models.types import Catalog, CatalogModel, ModelArtifact
from brasskey.output.next_step import NextStep

logger = logging.getLogger(__name__)

QUICK_START_MAX_DOWNLOAD_BYTES = 5 * 10**9  # about 5 GB (ADR 0002)
SAFETENSORS_QUALITY_ORDER = ("bf16", "fp16", "fp8", "awq-int4")

CachedBytes = Callable[[ModelArtifact], int]
# GPU backends that installed engines reported themselves (engines/registry.reported_gpu_backends).
GpuBackends = Mapping[EngineId, frozenset[str]]


class NoFitError(BrasskeyError):
    default_code = "BK-E-NO-FIT"


class UnsupportedEngineError(BrasskeyError):
    default_code = "BK-E-ENGINE-UNSUPPORTED"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunRequest(_Frozen):
    model: str | None = Field(default=None, description="Catalog id or hf:org/name[:QUANT].")
    engine: EngineId | None = None
    context_length: int = Field(default=8192, ge=MIN_CONTEXT)
    concurrency: int = Field(default=1, ge=1)


class RunPlan(_Frozen):
    request: RunRequest
    model: CatalogModel
    engine: EngineId
    fit: FitEstimate
    reasons: tuple[str, ...]
    bigger_options: tuple[FitEstimate, ...] = ()
    download: DownloadEstimate | None = None
    install: EngineInstall | None = Field(
        default=None, description="The detected engine installation (None: not installed)."
    )

    def with_download(self, download: DownloadEstimate) -> "RunPlan":
        return self.model_copy(update={"download": download})

    def with_install(self, install: EngineInstall | None) -> "RunPlan":
        return self.model_copy(update={"install": install})


@dataclass(frozen=True, slots=True)
class _EngineResult:
    best: FitEstimate | None
    tried: tuple[FitEstimate, ...]
    blocked: tuple[str, ...]  # why some artifacts were skipped


def _gib(size: int) -> str:
    return f"{size / GIB:.1f} GiB"


def _hardware_label(hw: HardwareReport) -> str:
    if hw.accelerators_of(AcceleratorKind.APPLE_GPU):
        return "Apple Silicon"
    if hw.accelerators_of(AcceleratorKind.NVIDIA_GPU):
        return "NVIDIA GPUs on Linux"
    return "CPU-only machines"


def _ordered_artifacts(engine: EngineId, artifacts: Sequence[ModelArtifact]) -> list[ModelArtifact]:
    if engine not in (EngineId.VLLM, EngineId.SGLANG):
        return list(artifacts)
    order = {name: index for index, name in enumerate(SAFETENSORS_QUALITY_ORDER)}
    return sorted(artifacts, key=lambda artifact: order.get(artifact.quantization, len(order)))


def _fit_for_engine(
    model: CatalogModel,
    engine: EngineId,
    hw: HardwareReport,
    request: RunRequest,
    *,
    download_limit: int | None = None,
    cached_bytes: CachedBytes = lambda _artifact: 0,
    backends: GpuBackends | None = None,
) -> _EngineResult:
    tried: list[FitEstimate] = []
    gpu_backends = (backends or {}).get(engine)  # None: unknown, assume the machine's GPU
    blocked: list[str] = []
    for artifact in _ordered_artifacts(engine, model.artifacts_for(engine)):
        quant = check_quantization(engine, artifact.quantization, hw)
        if not quant.supported:
            blocked.extend(quant.reasons)
            continue
        remaining = artifact.download_bytes - cached_bytes(artifact)
        if download_limit is not None and remaining > download_limit:
            continue
        fit = estimate_fit(
            model,
            artifact,
            engine,
            hw,
            context_length=request.context_length,
            concurrency=request.concurrency,
            gpu_backends=gpu_backends,
        )
        if quant.notes:
            fit = fit.model_copy(update={"notes": fit.notes + quant.notes})
        tried.append(fit)
    fitting = [fit for fit in tried if fit.fits]
    comfortable = [fit for fit in fitting if fit.comfortable]
    best = comfortable[0] if comfortable else (fitting[0] if fitting else None)
    return _EngineResult(best, tuple(tried), tuple(blocked))


def _engines(
    request: RunRequest,
    hw: HardwareReport,
    runnable: Collection[EngineId] | None,
    installed: Collection[EngineId] | None,
) -> list[EngineId]:
    """Engines to try, best first: supported here, startable by this build, installed first."""
    preferred = [
        engine
        for engine in engine_preference(hw)
        if check_engine(engine, hw).supported and (runnable is None or engine in runnable)
    ]
    if request.engine is not None:
        check = check_engine(request.engine, hw)
        usable = ", ".join(e.display_name for e in preferred) or "llama.cpp"
        if not check.supported:
            raise UnsupportedEngineError(
                f"{request.engine.display_name} can't run on this machine. "
                + " ".join(check.reasons),
                fix=f"Use {usable} here, or drop --engine to let Brasskey choose.",
            )
        if runnable is not None and request.engine not in runnable:
            raise UnsupportedEngineError(
                f"This pre-alpha build of Brasskey can't start {request.engine.display_name} yet.",
                fix=f"Use {usable} for now, or drop --engine.",
            )
        return [request.engine]
    if installed is not None:
        preferred.sort(key=lambda engine: engine not in installed)  # stable: keeps preference
    return preferred


def _engine_reason(engine: EngineId, request: RunRequest, hw: HardwareReport) -> str:
    if request.engine is not None:
        return f"{engine.display_name}: you asked for it."
    return (
        f"{engine.display_name}: Brasskey's choice on {_hardware_label(hw)} among the engines it "
        "can start and you have installed (a fixed rule until measured results are available)."
    )


def _artifact_reason(fit: FitEstimate, model: CatalogModel) -> str:
    quant = fit.artifact.quantization
    if not model.verified:
        return f"{quant}: the file chosen from {fit.artifact.repo_id}."
    if len(model.artifacts_for(fit.engine)) == 1:
        return f"{quant}: the {fit.engine.display_name} version in the catalog."
    if fit.comfortable:
        return f"{quant}: the best-quality version that fits with room to spare."
    return f"{quant}: the best version that fits, though with little room to spare."


def _round_context(tokens: int) -> int:
    """A tidy context length at or below ``tokens`` (multiples of 1024 when large enough)."""
    return tokens - tokens % 1024 if tokens >= 2048 else tokens


def _smallest_need(results: Sequence[FitEstimate]) -> FitEstimate | None:
    return min(results, key=lambda fit: fit.required_per_device_bytes, default=None)


def _no_fit(
    model: CatalogModel, closest: FitEstimate | None, request: RunRequest, suggestion: str | None
) -> NoFitError:
    if closest is None:
        return NoFitError(
            f"{model.display_name} has no version that this machine's engines can run.",
            fix="Pick another model, or pass --engine to try a specific engine.",
        )
    need = closest.required_per_device_bytes
    budget = closest.budget.per_device_bytes
    per = " per GPU" if closest.budget.device_count > 1 else ""
    message = (
        f"{model.display_name} doesn't fit here: {closest.engine.display_name} would need about "
        f"{_gib(need)}{per} and can use {_gib(budget)}{per} ({closest.budget.explanation})."
    )
    if MIN_CONTEXT <= closest.max_context_that_fits < request.context_length:
        tokens = _round_context(closest.max_context_that_fits)
        return NoFitError(
            message,
            fix=f"Use a shorter context: --context {tokens} fits.",
            next_step=NextStep(
                command=f"brasskey run {model.id} --dry-run --context {tokens}",
                hint="check the shorter context",
            ),
        )
    if suggestion:
        return NoFitError(
            message,
            fix=f"Pick a smaller model, for example {suggestion}.",
            next_step=NextStep(command=f"brasskey run {suggestion} --dry-run", hint="check it"),
        )
    return NoFitError(message, fix="Pick a smaller model or use a machine with more memory.")


def _largest_fitting_model(
    catalog: Catalog,
    engines: Sequence[EngineId],
    hw: HardwareReport,
    request: RunRequest,
    *,
    backends: GpuBackends | None = None,
) -> str | None:
    ladder = sorted(
        (model for model in catalog.models if model.quick_start_rank is not None),
        key=lambda model: model.quick_start_rank or 0,
        reverse=True,
    )
    for model in ladder:
        for engine in engines:
            if _fit_for_engine(model, engine, hw, request, backends=backends).best is not None:
                return model.id
    return None


def _plan_named_model(
    model: CatalogModel,
    engines: Sequence[EngineId],
    hw: HardwareReport,
    request: RunRequest,
    catalog: Catalog,
    *,
    backends: GpuBackends | None = None,
) -> RunPlan:
    tried: list[FitEstimate] = []
    blocked: list[str] = []
    for engine in engines:
        if not model.artifacts_for(engine):
            blocked.append(
                f"{model.display_name} has no {ENGINE_FORMATS[engine].value} files for "
                f"{engine.display_name}."
            )
            continue
        result = _fit_for_engine(model, engine, hw, request, backends=backends)
        tried.extend(result.tried)
        blocked.extend(result.blocked)
        if result.best is not None:
            return RunPlan(
                request=request,
                model=model,
                engine=engine,
                fit=result.best,
                reasons=(
                    _engine_reason(engine, request, hw),
                    _artifact_reason(result.best, model),
                ),
            )
    if not tried and blocked:
        raise NoFitError(
            f"{model.display_name} can't run on this machine. " + " ".join(blocked),
            fix="Pick another model or engine for this hardware.",
        )
    suggestion = _largest_fitting_model(catalog, engines, hw, request, backends=backends)
    raise _no_fit(model, _smallest_need(tried), request, suggestion)


def _plan_quick_start(
    catalog: Catalog,
    engines: Sequence[EngineId],
    hw: HardwareReport,
    request: RunRequest,
    cached_bytes: CachedBytes,
    installed: Collection[EngineId] | None,
    *,
    backends: GpuBackends | None = None,
) -> RunPlan:
    """Prefer what can run right now: try installed engines first, then every engine."""
    first = [engine for engine in engines if installed is not None and engine in installed]
    if first and len(first) < len(engines):
        try:
            return _quick_start_with(catalog, first, hw, request, cached_bytes, backends=backends)
        except NoFitError:
            logger.debug("No quick-start fit with installed engines %s; trying all", first)
    return _quick_start_with(catalog, engines, hw, request, cached_bytes, backends=backends)


def _quick_start_with(
    catalog: Catalog,
    engines: Sequence[EngineId],
    hw: HardwareReport,
    request: RunRequest,
    cached_bytes: CachedBytes,
    *,
    backends: GpuBackends | None = None,
) -> RunPlan:
    ladder = sorted(
        (model for model in catalog.models if model.quick_start_rank is not None),
        key=lambda model: model.quick_start_rank or 0,
        reverse=True,
    )
    quick: list[tuple[CatalogModel, FitEstimate]] = []
    any_size: list[tuple[CatalogModel, FitEstimate]] = []
    for model in ladder:
        for engine in engines:
            limited = _fit_for_engine(
                model,
                engine,
                hw,
                request,
                download_limit=QUICK_START_MAX_DOWNLOAD_BYTES,
                cached_bytes=cached_bytes,
                backends=backends,
            )
            if limited.best is not None:
                quick.append((model, limited.best))
                break
        for engine in engines:
            unlimited = _fit_for_engine(model, engine, hw, request, backends=backends)
            if unlimited.best is not None:
                any_size.append((model, unlimited.best))
                break

    chosen = next((pair for pair in quick if pair[1].comfortable), None) or next(iter(quick), None)
    if chosen is None:
        if any_size:  # something fits, but only with a big download
            smallest_fitting = any_size[-1][0]
            raise NoFitError(
                "Every catalog model that fits here needs more than about 5 GB of downloading.",
                fix=f"Name one to plan it anyway, e.g. `brasskey run {smallest_fitting.id} "
                "--dry-run`.",
                next_step=NextStep(
                    command=f"brasskey run {smallest_fitting.id} --dry-run", hint="plan it"
                ),
            )
        # Nothing fits at all: explain with the smallest model (and offer a shorter context).
        smallest = ladder[-1]
        tried = [
            fit
            for engine in engines
            for fit in _fit_for_engine(smallest, engine, hw, request, backends=backends).tried
        ]
        raise _no_fit(smallest, _smallest_need(tried), request, suggestion=None)
    model, fit = chosen
    rank = model.quick_start_rank or 0
    bigger = tuple(f for m, f in any_size if (m.quick_start_rank or 0) > rank)
    reasons = (
        f"{model.display_name}: the most capable quick-start model that fits with at least "
        f"{HEADROOM_TARGET:.0%} of memory to spare and downloads in about 5 GB or less."
        if fit.comfortable
        else f"{model.display_name}: the most capable quick-start model that fits here.",
        _engine_reason(fit.engine, request, hw),
        _artifact_reason(fit, model),
    )
    return RunPlan(
        request=request,
        model=model,
        engine=fit.engine,
        fit=fit,
        reasons=reasons,
        bigger_options=bigger,
    )


def plan_run(
    request: RunRequest,
    hw: HardwareReport,
    catalog: Catalog,
    *,
    hub: HubClient | None = None,
    cached_bytes: CachedBytes = lambda _artifact: 0,
    runnable: Collection[EngineId] | None = None,
    installed: Collection[EngineId] | None = None,
    gpu_backends: GpuBackends | None = None,
) -> RunPlan:
    """Decide what ``brasskey run`` would start. Raises a plain-language error if nothing fits.

    ``runnable``: engines this build can start (None: all). ``installed``: engines detected on
    this machine, tried first (None: installation unknown, keep the default order).
    ``gpu_backends``: GPU backends installed engines reported themselves; an engine that reported
    none (a CPU-only llama.cpp) is planned for the CPU. Engines left out use the machine's GPU.
    """
    engines = _engines(request, hw, runnable, installed)
    logger.debug(
        "Planning model=%s engines=%s context=%d gpu_backends=%s",
        request.model or "<quick start>",
        [engine.value for engine in engines],
        request.context_length,
        gpu_backends,
    )
    if request.model is None:
        return _plan_quick_start(
            catalog, engines, hw, request, cached_bytes, installed, backends=gpu_backends
        )
    if request.model.startswith("hf:"):
        if hub is None:
            raise ModelSourceError("Looking up Hugging Face models needs network access.")
        model = resolve_hf_model(request.model, hub)
    else:
        model = find_model(catalog, request.model)
    return _plan_named_model(model, engines, hw, request, catalog, backends=gpu_backends)
