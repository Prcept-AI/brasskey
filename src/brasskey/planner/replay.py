# SPDX-License-Identifier: Apache-2.0
"""Turn a run plan into a brasskey.lock, and a lock back into a run plan (replay)."""

import logging
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime

from brasskey import __version__
from brasskey.config.intent import Intent
from brasskey.config.io import intent_sha256
from brasskey.config.lock import (
    Lock,
    LockedEngine,
    LockedEstimate,
    LockedHardware,
    LockedMeasurement,
    LockedModel,
    LockedServe,
)
from brasskey.engines.capabilities import check_engine, check_quantization
from brasskey.engines.types import EngineId
from brasskey.hardware.types import GIB, HardwareReport
from brasskey.models.fit import estimate_fit
from brasskey.models.types import CatalogModel
from brasskey.planner.select import NoFitError, RunPlan, RunRequest, UnsupportedEngineError

logger = logging.getLogger(__name__)


def hardware_summary(hw: HardwareReport) -> str:
    parts = [hw.cpu.model_name, f"{round(hw.memory.total_bytes / GIB)} GiB RAM"]
    names = [acc.name for acc in hw.accelerators]
    if names:
        count = len(names)
        parts.append(f"{count} x {names[0]}" if count > 1 else names[0])
    return ", ".join(parts)


def lock_from_plan(
    plan: RunPlan,
    hw: HardwareReport,
    *,
    intent: Intent | None,
    host: str,
    port: int | None,
    api_key_env: str | None = None,
    api_key_source: str | None = None,
    tls: bool = False,
    launch_args: Sequence[str] = (),
    model_file: str | None = None,
    startup_seconds: float | None = None,
    placeholders: Mapping[str, str] | None = None,
) -> Lock:
    """A lock for ``plan``. ``startup_seconds`` is set once the engine actually started.

    Machine-specific text in the launch arguments (the model file, cache folders) becomes a
    placeholder, so locks hold no absolute paths (ADR 0006).
    """
    replacements = sorted((placeholders or {}).items(), key=lambda item: -len(item[0]))

    def portable(arg: str) -> str:
        if arg == model_file:
            return "{model}"
        for text, placeholder in replacements:
            arg = arg.replace(text, placeholder)
        return arg

    args = tuple(portable(arg) for arg in launch_args[1:])
    model = plan.model
    fit = plan.fit
    now = datetime.now(UTC)
    return Lock(
        generated_by=f"brasskey {__version__}",
        generated_at=now,
        intent_sha256=intent_sha256(intent) if intent is not None else None,
        hardware=LockedHardware(fingerprint=hw.fingerprint(), summary=hardware_summary(hw)),
        model=LockedModel(
            id=model.id,
            display_name=model.display_name,
            family=model.family,
            license=model.license,
            source_repo=model.source_repo,
            verified=model.verified,
            architecture=model.architecture,
        ),
        artifact=fit.artifact,
        engine=LockedEngine(
            id=plan.engine,
            version=plan.install.version if plan.install else None,
            runtime=plan.install.runtime if plan.install else None,
            image=plan.install.image if plan.install else None,
            args=args,
        ),
        serve=LockedServe(
            host=host,
            port=port,
            context_length=fit.context_length,
            concurrency=fit.concurrency,
            api_key_env=api_key_env,
            api_key_source=api_key_source,
            tls=tls,
        ),
        estimate=LockedEstimate(
            required_per_device_bytes=fit.required_per_device_bytes,
            budget_per_device_bytes=fit.budget.per_device_bytes,
            devices_used=fit.devices_used,
            headroom_ratio=fit.headroom_ratio,
        ),
        measured=LockedMeasurement(
            started=startup_seconds is not None,
            startup_seconds=round(startup_seconds, 2) if startup_seconds is not None else None,
            measured_at=now if startup_seconds is not None else None,
        ),
    )


def plan_from_lock(
    lock: Lock, hw: HardwareReport, *, runnable: Collection[EngineId] | None = None
) -> RunPlan:
    """Replay a lock on this machine: same files and engine; memory re-estimated here."""
    engine = lock.engine.id
    check = check_engine(engine, hw)
    if not check.supported:
        raise UnsupportedEngineError(
            f"brasskey.lock uses {engine.display_name}, which can't run here. "
            + " ".join(check.reasons),
            fix="Delete brasskey.lock to plan again for this machine.",
        )
    if runnable is not None and engine not in runnable:
        raise UnsupportedEngineError(
            f"brasskey.lock uses {engine.display_name}, which this build can't start yet.",
            fix="Update Brasskey, or delete brasskey.lock to plan again.",
        )
    quant = check_quantization(engine, lock.artifact.quantization, hw)
    if not quant.supported:
        raise UnsupportedEngineError(
            " ".join(quant.reasons), fix="Delete brasskey.lock to plan again for this machine."
        )
    model = CatalogModel(
        id=lock.model.id,
        display_name=lock.model.display_name,
        family=lock.model.family,
        license=lock.model.license,
        source_repo=lock.model.source_repo,
        verified=lock.model.verified,
        architecture=lock.model.architecture,
        artifacts=(lock.artifact,),
    )
    fit = estimate_fit(
        model,
        lock.artifact,
        engine,
        hw,
        context_length=lock.serve.context_length,
        concurrency=lock.serve.concurrency,
    )
    if not fit.fits:
        raise NoFitError(
            f"The model in brasskey.lock ({model.display_name}) doesn't fit this machine: it needs "
            f"about {fit.required_per_device_bytes / GIB:.1f} GiB and "
            f"{engine.display_name} can use {fit.budget.per_device_bytes / GIB:.1f} GiB.",
            fix="Run it on the machine it was locked on, or delete brasskey.lock to plan again.",
        )
    reasons = [
        f"Replaying brasskey.lock (written {lock.generated_at:%Y-%m-%d} by {lock.generated_by})."
    ]
    if hw.fingerprint() != lock.hardware.fingerprint:
        reasons.append(
            f"This machine differs from the one in the lock ({lock.hardware.summary}); the "
            "memory estimate was recomputed here."
        )
    logger.debug("Replaying lock: %s on %s", model.id, engine.value)
    return RunPlan(
        request=RunRequest(
            model=model.id,
            engine=engine,
            context_length=lock.serve.context_length,
            concurrency=lock.serve.concurrency,
        ),
        model=model,
        engine=engine,
        fit=fit,
        reasons=tuple(reasons),
    )
