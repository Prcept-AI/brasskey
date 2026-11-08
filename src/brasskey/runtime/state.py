# SPDX-License-Identifier: Apache-2.0
"""State files for running servers, so ``doctor`` (and later ``doctor --security``) can find them.

One private JSON file per server in ``<state dir>/servers/``. A record is live only if its process
still exists *and* started at the recorded time, which guards against process-id reuse; stale
records (for example after a crash or ``kill -9``) are removed when listed.
"""

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, ValidationError

from brasskey.paths import BrasskeyPaths, ensure_private_dir, open_private_file

logger = logging.getLogger(__name__)


class ServerRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    pid: int
    process_started_at: float  # psutil create_time(), to detect pid reuse
    engine: str
    engine_version: str
    model_id: str
    host: str
    port: int
    engine_port: int | None = None  # the engine's private loopback port behind the proxy
    tls: bool = False
    base_url: str
    auth: Literal["api_key", "none"]
    started_at: datetime
    log_path: str
    brasskey_version: str


def process_start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _record_path(paths: BrasskeyPaths, server_id: str) -> Path:
    return paths.servers_dir / f"{server_id}.json"


def write_record(paths: BrasskeyPaths, record: ServerRecord) -> Path:
    ensure_private_dir(paths.servers_dir)
    path = _record_path(paths, record.id)
    fd = open_private_file(path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(record.model_dump_json(indent=2))
    return path


def remove_record(paths: BrasskeyPaths, server_id: str) -> None:
    _record_path(paths, server_id).unlink(missing_ok=True)


def list_records(
    paths: BrasskeyPaths, *, start_time: Callable[[int], float | None] = process_start_time
) -> list[ServerRecord]:
    """Live servers; stale or unreadable records are deleted."""
    if not paths.servers_dir.is_dir():
        return []
    live: list[ServerRecord] = []
    for path in sorted(paths.servers_dir.glob("*.json")):
        try:
            record = ServerRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError):
            logger.debug("Removing unreadable server record %s", path.name)
            path.unlink(missing_ok=True)
            continue
        started = start_time(record.pid)
        if started is None or abs(started - record.process_started_at) > 1.0:
            logger.debug("Removing stale server record %s (pid %d)", path.name, record.pid)
            path.unlink(missing_ok=True)
            continue
        live.append(record)
    return live
