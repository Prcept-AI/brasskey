# SPDX-License-Identifier: Apache-2.0
"""Start, watch and stop engine processes (shared by every adapter).

- Each engine runs in its own session and process group, so stopping it also stops anything it
  started, and a Ctrl-C in the terminal reaches Brasskey (which then stops the engine cleanly)
  rather than killing the engine mid-write.
- Output goes to a private log file (mode 0600) in Brasskey's log folder.
- Secrets are copied from Brasskey's environment into the engine's environment by name; they are
  never placed in argv (visible to other users via ``ps``) or written anywhere.
- Stop sends SIGTERM to the group, waits, then SIGKILL. Containers are stopped first with the
  adapter's stop command (``docker stop``): signals reach only the Docker client, and a killed
  client would leave the container running.
"""

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx2

from brasskey.engines.base import (
    EngineAdapter,
    EngineNotInstalledError,
    EngineStartError,
    HealthStatus,
    LaunchSpec,
)
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandRunner, run_command
from brasskey.paths import BrasskeyPaths, ensure_private_dir, open_private_file

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 40
POLL_INTERVAL_S = 0.25
STOP_GRACE_S = 10.0
STOP_COMMAND_EXIT_WAIT_S = 10.0  # after `docker stop`, how long the client may take to exit


@dataclass
class ServerHandle:
    id: str
    engine_name: str
    spec: LaunchSpec
    process: subprocess.Popen[bytes]
    log_path: Path
    started_monotonic: float = field(default_factory=time.monotonic)
    run_dir: Path | None = None

    @property
    def pid(self) -> int:
        return self.process.pid


class ProcessSupervisor:
    def __init__(
        self,
        paths: BrasskeyPaths,
        *,
        env: Mapping[str, str] | None = None,
        run: CommandRunner = run_command,
    ) -> None:
        self.paths = paths
        self.env = dict(os.environ if env is None else env)
        self._run = run

    def _environment(self, spec: LaunchSpec) -> dict[str, str]:
        env = {**self.env, **spec.env}
        for engine_var, source_var in spec.secret_env.items():
            value = self.env.get(source_var)
            if not value:
                raise BrasskeyError(
                    f"The API key variable {source_var} is not set.",
                    fix=f"Export it first, e.g. `export {source_var}=...`, then run again.",
                    code="BK-E-API-KEY",
                )
            env[engine_var] = value
        return env

    def _make_run_dir(self, spec: LaunchSpec, server_id: str) -> Path | None:
        """A private working folder with the links the engine expects (removed on stop)."""
        if not spec.cwd_links:
            return None
        run_dir = ensure_private_dir(self.paths.state_dir / "runs" / server_id)
        for name, target in spec.cwd_links.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise BrasskeyError(f"Unsafe link name for the engine's folder: {name!r}.")
            link = run_dir / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        return run_dir

    def start(self, spec: LaunchSpec, *, server_id: str, engine_name: str) -> ServerHandle:
        log_path = ensure_private_dir(self.paths.log_dir) / f"{server_id}.log"
        env = self._environment(spec)
        run_dir = self._make_run_dir(spec, server_id)
        fd = open_private_file(log_path)
        try:
            # argv comes from an adapter (fixed flags + paths Brasskey chose), never a shell string.
            process = subprocess.Popen(  # noqa: S603
                list(spec.argv),
                stdin=subprocess.DEVNULL,
                stdout=fd,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=run_dir,
                start_new_session=True,
                close_fds=True,
            )
        except FileNotFoundError as error:
            self._remove_run_dir(run_dir)
            raise EngineNotInstalledError(
                f"{engine_name} isn't installed (couldn't run {spec.argv[0]}).",
                fix="Install it and run again; `brasskey doctor` shows what is installed.",
            ) from error
        finally:
            os.close(fd)
        logger.debug("Started %s pid=%d log=%s", engine_name, process.pid, log_path)
        return ServerHandle(server_id, engine_name, spec, process, log_path, run_dir=run_dir)

    @staticmethod
    def _remove_run_dir(run_dir: Path | None) -> None:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)  # removes the links, never their targets

    def log_tail(self, handle: ServerHandle, lines: int = LOG_TAIL_LINES) -> list[str]:
        try:
            text = handle.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    def failure(
        self, handle: ServerHandle, adapter: EngineAdapter, exit_code: int | None, what: str
    ) -> BrasskeyError:
        """A plain-language error for an engine that failed or stopped, pointing at its log."""
        tail = self.log_tail(handle)
        explained = adapter.explain_failure(exit_code, tail)
        log_hint = f"Full log: {handle.log_path}"
        if explained is not None:
            fix = f"{explained.fix} {log_hint}" if explained.fix else log_hint
            return EngineStartError(explained.message, fix=fix)
        last = next((line.strip() for line in reversed(tail) if line.strip()), "")
        detail = f" Last log line: {last}" if last else ""
        return EngineStartError(f"{handle.engine_name} {what}.{detail}", fix=log_hint)

    def wait_until_ready(
        self,
        handle: ServerHandle,
        adapter: EngineAdapter,
        *,
        client: httpx2.Client,
        timeout_s: float | None = None,
        on_tick: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> float:
        """Seconds until the engine reported ready. Raises (and stops it) on failure or timeout."""
        limit = timeout_s if timeout_s is not None else handle.spec.startup_timeout_s
        started = clock()
        while True:
            exit_code = handle.process.poll()
            if exit_code is not None:
                raise self.failure(
                    handle, adapter, exit_code, f"stopped while starting (exit code {exit_code})"
                )
            status = adapter.health(handle.spec, client)
            elapsed = clock() - started
            if status == HealthStatus.READY:
                logger.debug("%s ready after %.1fs", handle.engine_name, elapsed)
                return elapsed
            if status == HealthStatus.FAILED:
                self.stop(handle)
                raise self.failure(handle, adapter, None, "reported an error while starting")
            if elapsed > limit:
                self.stop(handle)
                raise self.failure(handle, adapter, None, f"wasn't ready after {limit:.0f}s")
            if on_tick is not None:
                on_tick(elapsed)
            sleep(POLL_INTERVAL_S)

    def stop(self, handle: ServerHandle, *, grace_s: float = STOP_GRACE_S) -> int | None:
        """Stop the engine and everything it started. Returns its exit code."""
        process = handle.process
        try:
            if process.poll() is not None:
                return process.returncode
            if handle.spec.stop_argv:
                code = self._stop_with_command(handle, grace_s)
                if code is not None:
                    return code
            self._signal_group(process.pid, signal.SIGTERM)
            try:
                return process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                logger.debug(
                    "%s ignored SIGTERM for %.0fs; sending SIGKILL", handle.engine_name, grace_s
                )
                self._signal_group(process.pid, signal.SIGKILL)
                return process.wait(timeout=5)
        finally:
            if process.poll() is not None:
                self._remove_run_dir(handle.run_dir)

    def _stop_with_command(self, handle: ServerHandle, grace_s: float) -> int | None:
        """Run the adapter's stop command; the client process then exits by itself."""
        spec = handle.spec
        out = self._run(list(spec.stop_argv), timeout_s=grace_s + 30)
        logger.debug(
            "%s stop command exited %d", handle.engine_name, out.returncode
        )  # e.g. 1 when the container is already gone; signals follow anyway
        try:
            return handle.process.wait(timeout=STOP_COMMAND_EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            return None

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):  # already gone
            os.killpg(pid, sig)  # the engine leads its own group (start_new_session)
