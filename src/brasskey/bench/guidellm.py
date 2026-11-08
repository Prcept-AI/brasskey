# SPDX-License-Identifier: Apache-2.0
"""GuideLLM (Apache-2.0, vLLM project) as Brasskey's load generator, run as a separate process.

Where it comes from (ADR 0002, ADR 0011):
- ``guidellm`` on PATH: used as is;
- otherwise a pinned, isolated environment run by ``uv tool run``: GuideLLM 0.8.1 with its
  dependencies resolved as of a fixed date and CPU-only PyTorch (GuideLLM only tokenizes; CUDA
  wheels would add gigabytes). Set up once after the user confirms; later runs start offline.

Each run (checked against GuideLLM 0.8.1):
- synthetic prompts of a fixed token shape, a fixed number of concurrent streams, a fixed
  duration and a seed;
- ``--metrics sample_size=0``: GuideLLM keeps no prompt or response text, only numbers;
- the API key reaches GuideLLM through ``GUIDELLM__SPEC__BACKEND__API_KEY`` (its settings
  variable), never argv; GuideLLM masks it in its report;
- it runs in a private folder (so no stray ``.env`` there can change it) and writes its JSON
  report there; Brasskey reads the report and deletes it.

The report parser checks GuideLLM's report format version and refuses unknown ones, so a
GuideLLM upgrade can't silently change what the numbers mean.
"""

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from brasskey.bench.models import RunMetrics, Workload
from brasskey.engines.base import ENGINE_BASE_ENV
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandRunner, run_command
from brasskey.paths import open_private_file
from brasskey.security.apikeys import ApiKey

logger = logging.getLogger(__name__)

GUIDELLM_VERSION = "0.8.1"
RESOLVE_CUTOFF = "2026-11-01T00:00:00Z"  # dependencies resolved as of this date: reproducible
UV_PYTHON = "3.12"
SETUP_NOTE = "about 300 MB to download, 1 GB on disk, one time"
REPORT_FORMAT_VERSION = 2  # GuideLLM 0.8 reports: metadata.version
KEY_ENV = "GUIDELLM__SPEC__BACKEND__API_KEY"
EXECUTABLE = "guidellm"
FALLBACK_DIRS = (Path.home() / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))
DOCS_URL = "https://github.com/vllm-project/guidellm"
RUN_GRACE_S = 300.0  # start-up, tokenizer loading and finishing in-flight requests


class BenchError(BrasskeyError):
    default_code = "BK-E-BENCH"


class ReportFormatError(BenchError):
    default_code = "BK-E-BENCH-REPORT"


@dataclass(frozen=True)
class GuidellmInstall:
    version: str
    runtime: Literal["process", "uvx"]
    executable: tuple[str, ...]
    needs_setup: bool = False
    setup_argv: tuple[str, ...] = ()
    setup_note: str | None = None

    @property
    def label(self) -> str:
        return f"GuideLLM {self.version}"


@dataclass(frozen=True)
class RunTarget:
    """One endpoint as GuideLLM sees it."""

    target: str  # server root, e.g. http://127.0.0.1:8000 (GuideLLM adds /v1/...)
    model: str
    tokenizer: str  # local folder or Hugging Face id
    tokenizer_local: bool
    api_key: ApiKey | None = field(default=None, repr=False)


def uv_base_command(uv: str, *, offline: bool = False) -> tuple[str, ...]:
    """``uv tool run`` with the pinned GuideLLM environment (its arguments come after)."""
    return (
        uv, "tool", "run", *(("--offline",) if offline else ()),
        "--managed-python", "--python", UV_PYTHON,
        "--exclude-newer", RESOLVE_CUTOFF,
        "--torch-backend", "cpu",
        "--from", f"guidellm=={GUIDELLM_VERSION}",
        EXECUTABLE,
    )  # fmt: skip


def parse_version(text: str) -> str | None:
    """``guidellm version: 0.8.1`` -> ``0.8.1``."""
    for line in text.splitlines():
        if "version" in line.lower() and ":" in line:
            value = line.split(":", 1)[1].strip()
            if value and value[0].isdigit():
                return value
    return None


def _find(name: str, which: Callable[[str], str | None], dirs: Sequence[Path]) -> str | None:
    found = which(name)
    if found:
        return found
    for directory in dirs:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return None


def detect_guidellm(
    *,
    run: CommandRunner = run_command,
    which: Callable[[str], str | None] = shutil.which,
    fallback_dirs: Sequence[Path] = FALLBACK_DIRS,
) -> GuidellmInstall | None:
    """GuideLLM on PATH, else the pinned uv environment (maybe not set up yet), else None."""
    on_path = _find(EXECUTABLE, which, fallback_dirs)
    if on_path is not None:
        out = run([on_path, "--version"], timeout_s=60)
        version = parse_version(out.stdout + out.stderr) if out.ok else None
        if version is not None:
            logger.debug("GuideLLM on PATH: %s %s", on_path, version)
            return GuidellmInstall(version=version, runtime="process", executable=(on_path,))
        logger.debug("%s --version failed (exit %d)", on_path, out.returncode)
    uv = _find("uv", which, fallback_dirs)
    if uv is None:
        return None
    ready = run([*uv_base_command(uv, offline=True), "--version"], timeout_s=60)
    if ready.ok and parse_version(ready.stdout + ready.stderr):
        logger.debug("Pinned GuideLLM environment ready")
        return GuidellmInstall(
            version=GUIDELLM_VERSION, runtime="uvx", executable=uv_base_command(uv)
        )
    logger.debug("Pinned GuideLLM environment not set up yet (uv exit %d)", ready.returncode)
    return GuidellmInstall(
        version=GUIDELLM_VERSION,
        runtime="uvx",
        executable=uv_base_command(uv),
        needs_setup=True,
        setup_argv=(*uv_base_command(uv), "--version"),
        setup_note=SETUP_NOTE,
    )


def install_hint() -> tuple[str, tuple[str, ...]]:
    return (
        "Install uv so Brasskey can set GuideLLM up automatically (or install GuideLLM).",
        ("brew install uv", f"or: uv tool install guidellm=={GUIDELLM_VERSION}"),
    )


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def build_argv(
    install: GuidellmInstall,
    target: RunTarget,
    workload: Workload,
    *,
    seed: int,
    duration_s: int,
    report_path: Path,
) -> list[str]:
    """The ``guidellm run`` command for one run (JSON options: paths may contain commas)."""
    return [
        *install.executable,
        "run",
        "--backend", _json({"kind": "openai_http", "target": target.target, "model": target.model}),
        "--tokenizer", _json({"kind": "huggingface_auto", "model": target.tokenizer}),
        "--data", _json({"kind": "synthetic_text", "prompt_tokens": workload.prompt_tokens,
                         "output_tokens": workload.output_tokens}),
        "--profile", _json({"kind": "concurrent", "streams": workload.concurrency}),
        "--constraint", _json({"kind": "max_duration", "seconds": duration_s}),
        "--seed", _json({"kind": "static", "value": seed}),
        "--metrics", _json({"kind": "generative", "sample_size": 0}),
        "--output", _json({"kind": "json", "path": str(report_path)}),
        "--disable-console",
    ]  # fmt: skip


def run_environment(target: RunTarget, base: Mapping[str, str]) -> dict[str, str]:
    """GuideLLM's environment: no Brasskey keys, no telemetry, and the target's key by name."""
    env = {
        name: value
        for name, value in base.items()
        if not name.startswith("GUIDELLM__") and "API_KEY" not in name
    }
    env.update(ENGINE_BASE_ENV)
    env["HF_HUB_OFFLINE"] = "1" if target.tokenizer_local else "0"
    if target.api_key is not None:
        env[KEY_ENV] = target.api_key.value
    return env


# --- the report --------------------------------------------------------------------------------


def _path(data: object, *keys: str) -> object:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            raise ReportFormatError(
                f"GuideLLM's report has no {'.'.join(keys)}.",
                fix=f"Use GuideLLM {GUIDELLM_VERSION} (let Brasskey run its pinned copy by "
                "removing other guidellm installs from PATH).",
            )
        current = cast(dict[str, object], current).get(key)
    return current


def _number(data: object, *keys: str) -> float:
    value = _path(data, *keys)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportFormatError(
            f"GuideLLM's report has no number at {'.'.join(keys)}.",
            fix=f"Use GuideLLM {GUIDELLM_VERSION}.",
        )
    return float(value)


def parse_report(text: str, *, seed: int) -> RunMetrics:
    """The numbers Brasskey uses from one GuideLLM JSON report (format version 2)."""
    try:
        data: object = json.loads(text)
    except ValueError as error:
        raise ReportFormatError("GuideLLM's report isn't valid JSON.") from error
    version = _path(data, "metadata", "version")
    if version != REPORT_FORMAT_VERSION:
        tool = _path(data, "metadata", "guidellm_version")
        raise ReportFormatError(
            f"GuideLLM {tool} writes report format {version}; this Brasskey reads format "
            f"{REPORT_FORMAT_VERSION} (GuideLLM 0.8).",
            fix=f"Use GuideLLM {GUIDELLM_VERSION}: remove other guidellm installs from PATH so "
            "Brasskey runs its pinned copy, or update Brasskey.",
        )
    benchmarks = _path(data, "benchmarks")
    if not isinstance(benchmarks, list) or len(cast(list[object], benchmarks)) != 1:
        raise ReportFormatError("GuideLLM's report should hold exactly one benchmark.")
    bench: object = cast(list[object], benchmarks)[0]
    metrics = _path(bench, "metrics")
    ok = ("successful",)
    return RunMetrics(
        seed=seed,
        duration_s=_number(bench, "duration"),
        successful=int(_number(metrics, "request_totals", "successful")),
        errored=int(_number(metrics, "request_totals", "errored")),
        incomplete=int(_number(metrics, "request_totals", "incomplete")),
        output_tokens_per_s=_number(metrics, "output_tokens_per_second", *ok, "mean"),
        requests_per_s=_number(metrics, "requests_per_second", *ok, "mean"),
        ttft_ms_p50=_number(metrics, "time_to_first_token_ms", *ok, "median"),
        ttft_ms_p95=_number(metrics, "time_to_first_token_ms", *ok, "percentiles", "p95"),
        itl_ms_p50=_number(metrics, "inter_token_latency_ms", *ok, "median"),
        latency_s_p50=_number(metrics, "request_latency", *ok, "median"),
        latency_s_p95=_number(metrics, "request_latency", *ok, "percentiles", "p95"),
    )


# --- one run ---------------------------------------------------------------------------------


_FAILURES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("401", "unauthorized", "invalid_api_key"),
        "The endpoint refused GuideLLM's requests (API key).",
        "Set the key's variable (BRASSKEY_API_KEY or --api-key-env), then run again.",
    ),
    (
        ("connecterror", "connection refused", "all connection attempts failed"),
        "GuideLLM couldn't reach the endpoint.",
        "Check that the server is still running (`brasskey doctor` lists Brasskey servers).",
    ),
    (
        ("can't load tokenizer", "tokenizer", "not a valid model identifier"),
        "GuideLLM couldn't load the tokenizer.",
        "Pass --tokenizer with the model's Hugging Face id or a local folder with tokenizer.json.",
    ),
    (
        ("no such option", "invalid value for", "unrecognized arguments"),
        "This GuideLLM version doesn't accept Brasskey's options.",
        f"Use GuideLLM {GUIDELLM_VERSION}: remove other guidellm installs from PATH so Brasskey "
        "runs its pinned copy.",
    ),
)


def explain_failure(log_tail: Sequence[str]) -> tuple[str, str] | None:
    text = "\n".join(log_tail).lower()
    for fragments, message, fix in _FAILURES:
        if any(fragment in text for fragment in fragments):
            return message, fix
    return None


def run_once(
    install: GuidellmInstall,
    target: RunTarget,
    workload: Workload,
    *,
    seed: int,
    duration_s: int,
    work_dir: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> RunMetrics:
    """One GuideLLM run; its raw report (which may hold request details) is deleted at once."""
    report = work_dir / f"report-{seed}.json"
    argv = build_argv(
        install, target, workload, seed=seed, duration_s=duration_s, report_path=report
    )
    environment = run_environment(target, os.environ if env is None else env)
    fd = open_private_file(log_path, append=True)  # one log for the whole bench
    try:
        logger.debug(
            "GuideLLM run: seed=%d duration=%ds target=%s", seed, duration_s, target.target
        )
        # argv is built here from fixed options and values Brasskey checked; never a shell.
        completed = subprocess.run(  # noqa: S603
            argv,
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
            env=environment,
            timeout=duration_s + RUN_GRACE_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise BenchError(
            f"GuideLLM didn't finish within {duration_s + RUN_GRACE_S:.0f}s.",
            fix=f"Check that the endpoint still answers. GuideLLM's log: {log_path}",
        ) from error
    except FileNotFoundError as error:
        raise BenchError(
            f"GuideLLM couldn't start ({argv[0]} not found).",
            fix="Install uv (`brew install uv`) and run again.",
        ) from error
    finally:
        os.close(fd)
    try:
        if completed.returncode != 0 or not report.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            explained = explain_failure(tail)
            message, fix = explained or (
                f"GuideLLM failed (exit code {completed.returncode}).",
                "See its log for details.",
            )
            raise BenchError(message, fix=f"{fix} GuideLLM's log: {log_path}")
        metrics = parse_report(report.read_text(encoding="utf-8"), seed=seed)
    finally:
        report.unlink(missing_ok=True)
    if metrics.successful == 0:
        raise BenchError(
            f"No request succeeded ({metrics.errored} failed).",
            fix=f"Check the endpoint with `curl`, then run again. GuideLLM's log: {log_path}",
        )
    return metrics
