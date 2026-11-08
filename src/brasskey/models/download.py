# SPDX-License-Identifier: Apache-2.0
"""Download estimates: size, what is already cached, disk space, and time at a measured speed.

The speed is measured, not assumed: a ranged request for the start of the model's largest file is
timed until 8 MiB arrive or 3 seconds pass, whichever comes first (checked against curl on the same
file). It is skipped offline and when it fails; the estimate then has no time.
"""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field

from brasskey import __version__
from brasskey.hardware.types import MIB
from brasskey.models.hub import HubClient
from brasskey.models.types import ModelArtifact

logger = logging.getLogger(__name__)

SAMPLE_BYTES = 8 * MIB
SAMPLE_SECONDS = 3.0  # keep dry runs quick on slow links
PROBE_TIMEOUT_S = 8.0
USER_AGENT = f"brasskey/{__version__}"

BandwidthProbe = Callable[[str], float | None]
DiskUsage = Callable[[Path], int]


class DownloadEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_bytes: int
    cached_bytes: int
    to_download_bytes: int
    bandwidth_bytes_per_s: float | None = None
    bandwidth_source: Literal["measured", "offline", "failed", "not-needed"]
    eta_seconds: float | None = None
    cache_dir: str
    disk_free_bytes: int | None = None
    fits_on_disk: bool | None = Field(default=None, description="None when disk space is unknown.")


def measure_bandwidth(
    url: str,
    *,
    client: httpx2.Client | None = None,
    sample_bytes: int = SAMPLE_BYTES,
    sample_seconds: float = SAMPLE_SECONDS,
    clock: Callable[[], float] = time.perf_counter,
) -> float | None:
    """Bytes per second for a ranged download of ``url``; None if it can't be measured."""
    owns_client = client is None
    active = client or httpx2.Client(
        follow_redirects=True,
        timeout=PROBE_TIMEOUT_S,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        headers = {"Range": f"bytes=0-{sample_bytes - 1}"}
        with active.stream("GET", url, headers=headers) as response:
            if response.status_code not in (200, 206):
                logger.debug("Bandwidth probe got HTTP %d", response.status_code)
                return None
            started = clock()  # after the headers: measures throughput, not latency
            received = 0
            elapsed = 0.0
            for chunk in response.iter_bytes():
                received += len(chunk)
                elapsed = clock() - started
                if received >= sample_bytes or elapsed >= sample_seconds:
                    break
    except httpx2.HTTPError as error:
        logger.debug("Bandwidth probe failed: %s", type(error).__name__)
        return None
    finally:
        if owns_client:
            active.close()
    if received < 256 * 1024 or elapsed <= 0:
        return None  # too little data for a meaningful rate
    rate = received / elapsed
    logger.debug("Bandwidth probe: %d bytes in %.2fs = %.1f MB/s", received, elapsed, rate / 1e6)
    return rate


def _disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def estimate_download(
    artifact: ModelArtifact,
    *,
    hub: HubClient,
    bandwidth: BandwidthProbe | None,
    disk_free: DiskUsage = _disk_free,
) -> DownloadEstimate:
    """Estimate the download for ``artifact``. ``bandwidth=None`` means offline."""
    total = artifact.download_bytes
    cached = min(hub.cached_bytes(artifact), total)
    remaining = total - cached
    cache_dir = hub.cache_dir()
    try:
        free: int | None = disk_free(_existing_parent(cache_dir))
    except OSError:
        free = None

    rate: float | None = None
    source: Literal["measured", "offline", "failed", "not-needed"]
    if remaining == 0:
        source = "not-needed"
    elif bandwidth is None:
        source = "offline"
    else:
        largest = max(artifact.files, key=lambda file: file.size_bytes)
        rate = bandwidth(hub.file_url(artifact.repo_id, artifact.revision, largest.path))
        source = "measured" if rate else "failed"
    return DownloadEstimate(
        total_bytes=total,
        cached_bytes=cached,
        to_download_bytes=remaining,
        bandwidth_bytes_per_s=rate,
        bandwidth_source=source,
        eta_seconds=round(remaining / rate, 1) if rate else (0.0 if remaining == 0 else None),
        cache_dir=str(cache_dir),
        disk_free_bytes=free,
        fits_on_disk=(remaining <= free) if free is not None else None,
    )
