# SPDX-License-Identifier: Apache-2.0
"""``brasskey bench``: benchmark an OpenAI-compatible endpoint with repeated runs.

GuideLLM generates the load (``brasskey.bench.guidellm``); Brasskey decides the workload, runs a
warm-up and repeated measured runs, reports every metric with a 95% interval, and, when given a
baseline (``--compare URL`` or ``--baseline FILE``), a PASS / FAIL / INCONCLUSIVE verdict with
exit codes 0 / 3 / 4 (ADR 0011). Numbers only: no prompt or response text is
kept anywhere.
"""

import logging
import secrets
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import httpx2
import typer
from pydantic import JsonValue, ValidationError
from rich.console import Group
from rich.text import Text

from brasskey import __version__
from brasskey.bench.analysis import compare, summarize
from brasskey.bench.endpoint import (
    brasskey_server,
    choose_model,
    parse_endpoint,
    served_models,
)
from brasskey.bench.guidellm import (
    BenchError,
    GuidellmInstall,
    RunTarget,
    detect_guidellm,
    install_hint,
    run_once,
)
from brasskey.bench.models import (
    BenchReport,
    EndpointResult,
    RunMetrics,
    Workload,
    metric_by_option,
)
from brasskey.bench.runner import Step, estimate_seconds, execute, schedule
from brasskey.bench.stats import Verdict
from brasskey.bench.tokenizer import TokenizerChoice, choose_tokenizer, fetch_tokenizer
from brasskey.cli.options import DebugFlag, JsonFlag, YesFlag, apply_flags
from brasskey.config.io import read_lock, write_lock
from brasskey.config.lock import LOCK_FILE, Lock, LockedBenchmark
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError, ExitCode
from brasskey.log import register_secret
from brasskey.models.catalog import load_catalog
from brasskey.models.hub import HubClient, RealHub
from brasskey.output.bench import bench_view, workload_text
from brasskey.output.confirm import confirm
from brasskey.output.next_step import NextStep
from brasskey.output.plan import format_duration, format_gb
from brasskey.output.render import stderr_console
from brasskey.output.result import CommandResult
from brasskey.output.settings import OutputSettings
from brasskey.paths import BrasskeyPaths, ensure_private_dir, resolve_paths
from brasskey.runtime.setup import run_setup
from brasskey.runtime.state import ServerRecord, list_records
from brasskey.security.apikeys import ApiKey, default_keyring_get, resolve_api_key

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 10.0
SETUP_TIMEOUT_S = 1800.0


class GoalOption(StrEnum):
    BETTER = "better"
    NOT_WORSE = "not-worse"


class MetricOption(StrEnum):
    OUTPUT_THROUGHPUT = "output-throughput"
    REQUEST_THROUGHPUT = "request-throughput"
    TTFT = "ttft"
    ITL = "itl"
    LATENCY = "latency"


# --- seams for tests ----------------------------------------------------------------------------


def make_hub() -> HubClient:
    return RealHub(offline=False)


def make_http_client() -> httpx2.Client:
    # trust_env=False: benchmark traffic and probes never go through a proxy from the environment.
    return httpx2.Client(timeout=HTTP_TIMEOUT_S, trust_env=False)


def detect_guidellm_install() -> GuidellmInstall | None:
    return detect_guidellm()


def run_guidellm_step(
    install: GuidellmInstall,
    target: RunTarget,
    workload: Workload,
    *,
    seed: int,
    duration_s: int,
    work_dir: Path,
    log_path: Path,
) -> RunMetrics:
    return run_once(
        install,
        target,
        workload,
        seed=seed,
        duration_s=duration_s,
        work_dir=work_dir,
        log_path=log_path,
    )


# The OS keyring (tests replace this so they never touch a real keychain).
keyring_reader = default_keyring_get


# --- helpers -------------------------------------------------------------------------------------


@contextmanager
def _status(settings: OutputSettings, message: str) -> Generator[None, None, None]:
    if settings.json:
        yield
        return
    with stderr_console().status(message):
        yield


def _key(env_name: str | None) -> ApiKey | None:
    key = resolve_api_key(env_name, keyring_get=keyring_reader)
    if key is not None:
        register_secret(key.value)  # redacted from any log line
    return key


def _workload(
    base: Workload,
    *,
    prompt_tokens: int | None,
    output_tokens: int | None,
    concurrency: int | None,
    duration: int | None,
    repeats: int | None,
    warmup: int | None,
) -> Workload:
    changes = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "duration_s": duration,
        "repeats": repeats,
        "warmup_s": warmup,
    }
    try:
        return Workload.model_validate(
            base.model_dump() | {key: value for key, value in changes.items() if value is not None}
        )
    except ValidationError as error:
        problem = error.errors()[0]
        raise BrasskeyError(
            f"Invalid benchmark setting {problem['loc'][0]}: {problem['msg']}.",
            code="BK-E-USAGE",
            exit_code=ExitCode.USAGE,
        ) from error


def _load_baseline(path: Path) -> BenchReport:
    try:
        return BenchReport.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchError(
            f"Can't read {path}.", fix="Pass a file written by `brasskey bench --save`."
        ) from error
    except ValidationError as error:
        raise BenchError(
            f"{path} isn't a Brasskey benchmark result (format brasskey.bench/v1).",
            fix="Pass a file written by `brasskey bench --save`.",
        ) from error


def _check_same_workload(saved: Workload, wanted: Workload, retry: str) -> None:
    fields = ("prompt_tokens", "output_tokens", "concurrency", "duration_s", "seed")
    different = [name for name in fields if getattr(saved, name) != getattr(wanted, name)]
    if different:
        raise BenchError(
            f"The saved baseline used a different workload ({', '.join(different)}).",
            fix="Drop those options so this run reuses the saved workload.",
            code="BK-E-USAGE",
            next_step=NextStep(command=retry, hint="reuse the saved workload"),
        )


def _label(record: ServerRecord | None) -> tuple[str | None, str | None]:
    """Engine name and version when the endpoint is a running Brasskey server."""
    if record is None:
        return None, None
    try:
        name = EngineId(record.engine).display_name
    except ValueError:
        name = record.engine
    return name, record.engine_version


def _confirm_question(
    endpoints: list[str],
    steps: list[Step],
    install: GuidellmInstall,
    tokenizer: TokenizerChoice,
) -> str:
    parts: list[str] = []
    if install.needs_setup:
        parts.append(f"set up {install.label} ({install.setup_note})")
    if tokenizer.download is not None:
        parts.append(f"download the tokenizer ({format_gb(tokenizer.download_bytes)})")
    total = estimate_seconds(steps)
    target = " and ".join(endpoints)
    parts.append(f"benchmark {target} for {format_duration(total)} ({len(steps)} runs)")
    question = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    return question[0].upper() + question[1:] + "?"


def _again(endpoint: str, *extra: str) -> str:
    return " ".join(["brasskey", "bench", endpoint, *extra])


# --- the command ---------------------------------------------------------------------------------


def bench_command(
    endpoint: Annotated[
        str,
        typer.Argument(
            help="Base URL of an OpenAI-compatible server, e.g. http://127.0.0.1:8000/v1",
            show_default=False,
        ),
    ],
    compare_url: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help="Baseline endpoint to measure now and compare against.",
            show_default=False,
        ),
    ] = None,
    baseline_file: Annotated[
        Path | None,
        typer.Option(
            "--baseline", help="Saved results (`--save`) to compare against.", show_default=False
        ),
    ] = None,
    metric: Annotated[
        MetricOption,
        typer.Option("--metric", help="The metric that decides the verdict."),
    ] = MetricOption.OUTPUT_THROUGHPUT,
    goal: Annotated[
        GoalOption,
        typer.Option("--goal", help="Pass when the candidate is better, or not worse."),
    ] = GoalOption.BETTER,
    margin: Annotated[
        float,
        typer.Option("--margin", min=0.0, max=0.5, help="Margin as a fraction (0.05 = 5%)."),
    ] = 0.0,
    repeats: Annotated[
        int | None, typer.Option("--repeats", min=1, help="Measured runs per endpoint [3].")
    ] = None,
    duration: Annotated[
        int | None, typer.Option("--duration", min=5, help="Seconds per run [30].")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", min=1, help="Concurrent requests [4].")
    ] = None,
    prompt_tokens: Annotated[
        int | None, typer.Option("--prompt-tokens", min=1, help="Prompt length in tokens [256].")
    ] = None,
    output_tokens: Annotated[
        int | None, typer.Option("--output-tokens", min=1, help="Answer length in tokens [128].")
    ] = None,
    warmup: Annotated[
        int | None,
        typer.Option("--warmup", min=0, help="Seconds of discarded warm-up per endpoint [10]."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name, when the endpoint serves several."),
    ] = None,
    tokenizer: Annotated[
        str | None,
        typer.Option("--tokenizer", help="Hugging Face id or folder (for non-catalog models)."),
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option("--api-key-env", help="Variable holding the endpoint's API key."),
    ] = None,
    compare_api_key_env: Annotated[
        str | None,
        typer.Option("--compare-api-key-env", help="Variable holding the baseline's API key."),
    ] = None,
    save: Annotated[
        Path | None,
        typer.Option("--save", help="Write the results (numbers only) to this JSON file."),
    ] = None,
    lock: Annotated[
        bool,
        typer.Option("--lock", help=f"Append the results to {LOCK_FILE} in this folder."),
    ] = False,
    json_output: JsonFlag = False,
    debug: DebugFlag = False,
    yes: YesFlag = False,
) -> CommandResult:
    """Benchmark an endpoint and report results with confidence intervals."""
    settings = apply_flags("bench", json_output=json_output, debug=debug, yes=yes)
    # Only the scheme is logged: URLs can carry credentials in their userinfo part.
    logger.debug("bench requested (endpoint scheme=%s)", endpoint.split(":", 1)[0])
    if compare_url is not None and baseline_file is not None:
        raise BrasskeyError(
            "--compare and --baseline both name a baseline.",
            fix="Use one: --compare URL measures it now, --baseline FILE uses saved results.",
            code="BK-E-USAGE",
            exit_code=ExitCode.USAGE,
        )
    primary = metric_by_option(metric.value)
    candidate_endpoint = parse_endpoint(endpoint)
    baseline_endpoint = parse_endpoint(compare_url) if compare_url is not None else None
    saved = _load_baseline(baseline_file) if baseline_file is not None else None
    workload = _workload(
        saved.workload if saved is not None else Workload(),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        duration=duration,
        repeats=repeats,
        warmup=warmup,
    )
    if saved is not None:
        _check_same_workload(
            saved.workload, workload, _again(endpoint, "--baseline", str(baseline_file))
        )

    candidate_key = _key(api_key_env)
    baseline_key = _key(compare_api_key_env or api_key_env) if baseline_endpoint else None
    paths = resolve_paths()
    records = list_records(paths)
    with make_http_client() as client:
        candidate_model = choose_model(
            candidate_endpoint, served_models(client, candidate_endpoint, candidate_key), model
        )
        baseline_model = None
        if baseline_endpoint is not None:
            served = served_models(client, baseline_endpoint, baseline_key)
            wanted = candidate_model if candidate_model in served else None
            baseline_model = choose_model(baseline_endpoint, served, wanted)

    if lock:
        _lock_for(candidate_model)
    if save is not None and not save.parent.resolve().is_dir():
        raise BrasskeyError(
            f"Can't save to {save}: the folder {save.parent} doesn't exist.",
            code="BK-E-USAGE",
            exit_code=ExitCode.USAGE,
        )

    hub = make_hub()
    tokenizer_choice = choose_tokenizer(candidate_model, tokenizer, catalog=load_catalog(), hub=hub)
    install = detect_guidellm_install()
    if install is None:
        summary, commands = install_hint()
        raise BenchError(
            "GuideLLM isn't available, so Brasskey can't generate load.",
            fix=f"{summary} Run: {commands[0]}",
            code="BK-E-BENCH-TOOL",
            next_step=NextStep(command=commands[0], hint="install uv"),
        )

    roles = ["candidate"] + (["baseline"] if baseline_endpoint is not None else [])
    steps = schedule(workload, roles)
    shown = [candidate_endpoint.api_base] + (
        [baseline_endpoint.api_base] if baseline_endpoint is not None else []
    )
    if not settings.json:
        stderr_console().print(
            f"Plan: {len(steps)} runs ({workload_text(workload)}), about "
            f"{format_duration(estimate_seconds(steps)).removeprefix('about ')}; tokenizer "
            f"{tokenizer_choice.description}; load from {install.label}."
        )
    question = _confirm_question(shown, steps, install, tokenizer_choice)
    if not confirm(question, settings=settings):
        raise BrasskeyError(
            "Cancelled: nothing was run.",
            code="BK-E-CANCELLED",
            next_step=NextStep(command=_again(endpoint, "--repeats", "2"), hint="a shorter run"),
        )

    if install.needs_setup:
        with _status(settings, f"Setting up {install.label} ({install.setup_note})..."):
            run_setup(install.setup_argv, name=install.label, timeout_s=SETUP_TIMEOUT_S)
        install = detect_guidellm_install() or install
    if tokenizer_choice.download is not None:
        with _status(settings, "Downloading the tokenizer..."):
            tokenizer_choice = fetch_tokenizer(tokenizer_choice, hub)

    targets = {
        "candidate": RunTarget(
            target=candidate_endpoint.root,
            model=candidate_model,
            tokenizer=tokenizer_choice.source,
            tokenizer_local=tokenizer_choice.local,
            api_key=candidate_key,
        )
    }
    if baseline_endpoint is not None and baseline_model is not None:
        # The same tokenizer and seeds for both: identical prompts, a paired comparison.
        targets["baseline"] = RunTarget(
            target=baseline_endpoint.root,
            model=baseline_model,
            tokenizer=tokenizer_choice.source,
            tokenizer_local=tokenizer_choice.local,
            api_key=baseline_key,
        )
    results = _measure(install, targets, workload, steps, paths, settings)

    engine, version = _label(brasskey_server(candidate_endpoint, records))
    candidate = EndpointResult(
        url=candidate_endpoint.api_base,
        model=candidate_model,
        engine=engine,
        engine_version=version,
        runs=tuple(results["candidate"]),
        metrics=summarize(results["candidate"]),
    )
    baseline: EndpointResult | None = None
    if baseline_endpoint is not None and baseline_model is not None:
        engine, version = _label(brasskey_server(baseline_endpoint, records))
        baseline = EndpointResult(
            url=baseline_endpoint.api_base,
            model=baseline_model,
            engine=engine,
            engine_version=version,
            runs=tuple(results["baseline"]),
            metrics=summarize(results["baseline"]),
        )
    elif saved is not None:
        baseline = saved.candidate
    comparison = (
        compare(
            baseline,
            candidate,
            goal="better" if goal == GoalOption.BETTER else "not-worse",
            margin=margin,
            primary=primary.key,
            baseline_source="file" if saved is not None else "endpoint",
        )
        if baseline is not None
        else None
    )
    report = BenchReport(
        created_at=datetime.now(UTC),
        brasskey_version=__version__,
        tool=f"guidellm {install.version}",
        workload=workload,
        candidate=candidate,
        baseline=baseline,
        comparison=comparison,
    )
    notes: list[str] = []
    if save is not None:
        save.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        notes.append(f"Wrote {save}.")
    if lock:
        notes.append(_append_to_lock(report))

    return _result(report, endpoint, save, notes)


def _measure(
    install: GuidellmInstall,
    targets: dict[str, RunTarget],
    workload: Workload,
    steps: list[Step],
    paths: BrasskeyPaths,
    settings: OutputSettings,
) -> dict[str, list[RunMetrics]]:
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"
    work_dir = ensure_private_dir(paths.state_dir / "bench" / run_id)
    log_path = ensure_private_dir(paths.log_dir) / f"bench-{run_id}.log"
    total = len(steps)

    def run(step: Step) -> RunMetrics:
        what = (
            f"warm-up on the {step.role}"
            if not step.measured
            else (f"run {step.repeat} of {workload.repeats} on the {step.role}")
        )
        with _status(settings, f"Benchmarking: {what} ({step.duration_s} s)..."):
            return run_guidellm_step(
                install,
                targets[step.role],
                workload,
                seed=step.seed,
                duration_s=step.duration_s,
                work_dir=work_dir,
                log_path=log_path,
            )

    def progress(number: int, step: Step) -> None:
        logger.debug("Step %d of %d: %s seed=%d", number, total, step.role, step.seed)

    try:
        return execute(steps, run, on_step=progress)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)  # never leave GuideLLM's raw reports behind


def _lock_for(model: str) -> Lock:
    """The lock --lock will extend; checked before any run so a mismatch costs no time."""
    current = read_lock(Path.cwd())
    if current is None:
        raise BenchError(
            f"There is no {LOCK_FILE} in this folder to add the results to.",
            fix="Run `brasskey run --save` here first, or drop --lock.",
        )
    if current.model.id != model:
        raise BenchError(
            f"{LOCK_FILE} is for {current.model.id}, but the endpoint serves {model}.",
            fix="Benchmark the server that lock describes, or drop --lock.",
        )
    return current


def _append_to_lock(report: BenchReport) -> str:
    folder = Path.cwd()
    current = _lock_for(report.candidate.model)
    entry = LockedBenchmark(
        measured_at=report.created_at,
        tool=report.tool,
        model=report.candidate.model,
        workload=report.workload,
        metrics=report.candidate.metrics,
    )
    write_lock(folder, current.model_copy(update={"benchmarks": (*current.benchmarks, entry)}))
    return f"Added the results to {LOCK_FILE}."


def _result(
    report: BenchReport, endpoint: str, save: Path | None, notes: list[str]
) -> CommandResult:
    data: dict[str, JsonValue] = report.model_dump(mode="json")
    data["notes"] = list(notes)
    comparison = report.comparison
    exit_code = ExitCode.OK
    if comparison is not None:
        exit_code = {
            Verdict.PASS: ExitCode.OK,
            Verdict.FAIL: ExitCode.CHECK_FAILED,
            Verdict.INCONCLUSIVE: ExitCode.INCONCLUSIVE,
        }[comparison.verdict]
        data["verdict"] = comparison.verdict.value
    if comparison is not None and comparison.verdict == Verdict.INCONCLUSIVE:
        more = str(report.workload.repeats + 2)
        next_step = NextStep(
            command=_again(endpoint, "--repeats", more, "..."),
            hint="more runs narrow the interval",
        )
    elif save is None:
        next_step = NextStep(
            command=_again(endpoint, "--save", "bench.json"), hint="keep results to compare later"
        )
    else:
        next_step = NextStep(
            command=f"brasskey bench <other endpoint> --baseline {save}",
            hint="compare another setup against these results",
        )
    view = Group(bench_view(report), *(Text(note) for note in notes))
    return CommandResult(
        command="bench", data=data, next_step=next_step, human=view, exit_code=exit_code
    )
