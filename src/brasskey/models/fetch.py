# SPDX-License-Identifier: Apache-2.0
"""Download an artifact into the Hugging Face cache and verify it before any engine sees it.

- Files come from the pinned revision only; downloads resume after an interruption (the hub keeps
  ``.incomplete`` files).
- Files fetched now are hashed and compared with the catalog's sha256 (supply-chain rule:
  never run unverified weights). Files already in the cache are checked cheaply: the hub stores
  large files under their sha256, so the cached blob's name must match.
- The cache is shared with engines and other Hugging Face tools, so nothing is downloaded twice.
"""

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from brasskey.engines.base import LocalModel
from brasskey.errors import BrasskeyError
from brasskey.models.hub import HubClient
from brasskey.models.types import ArtifactFile, ModelArtifact

logger = logging.getLogger(__name__)

HASH_CHUNK = 8 * 1024 * 1024

# (bytes done, bytes total) for the whole artifact; called from download threads.
ProgressCallback = Callable[[int, int], None]


class DownloadError(BrasskeyError):
    default_code = "BK-E-DOWNLOAD"


class IntegrityError(BrasskeyError):
    default_code = "BK-E-INTEGRITY"


def sha256_file(path: Path, on_bytes: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
            if on_bytes is not None:
                on_bytes(len(chunk))
    return digest.hexdigest()


def _cached_blob_matches(path: Path, expected: str) -> bool:
    """The hub names cached large files after their sha256; a symlink target proves which blob."""
    return path.resolve().name == expected


def _verify(
    file: ArtifactFile, path: Path, *, fresh: bool, on_bytes: Callable[[int], None] | None
) -> None:
    if path.stat().st_size != file.size_bytes:
        raise IntegrityError(
            f"{file.path} has {path.stat().st_size:,} bytes; the catalog expects "
            f"{file.size_bytes:,}.",
            fix="Delete it from the Hugging Face cache and run again to re-download it.",
        )
    if file.sha256 is None:
        return
    if not fresh and _cached_blob_matches(path, file.sha256):
        return
    actual = sha256_file(path, on_bytes)
    if actual != file.sha256:
        raise IntegrityError(
            f"{file.path} doesn't match the pinned checksum (expected {file.sha256[:12]}..., got "
            f"{actual[:12]}...). Brasskey won't run it.",
            fix="Delete it from the Hugging Face cache and run again; if it repeats, report it.",
        )


def fetch_artifact(
    artifact: ModelArtifact,
    *,
    hub: HubClient,
    on_progress: ProgressCallback | None = None,
    on_verify: ProgressCallback | None = None,
) -> LocalModel:
    """Download (if needed) and verify every file of ``artifact``; return where they are."""
    total = artifact.download_bytes
    done_before = 0
    paths: list[Path] = []
    for file in artifact.files:
        cached = hub.cached_path(artifact.repo_id, artifact.revision, file.path)
        fresh = cached is None
        if cached is None:
            logger.debug("Downloading %s (%d bytes)", file.path, file.size_bytes)
            base = done_before

            def report(done: int, base: int = base) -> None:
                if on_progress is not None:
                    on_progress(base + done, total)

            path = hub.download_file(
                artifact.repo_id, artifact.revision, file.path, file.size_bytes, report
            )
        else:
            path = cached
        done_before += file.size_bytes
        if on_progress is not None:
            on_progress(done_before, total)

        checked = 0

        def verified(chunk: int, size: int = file.size_bytes) -> None:
            nonlocal checked
            checked += chunk
            if on_verify is not None:
                on_verify(checked, size)

        _verify(file, path, fresh=fresh, on_bytes=verified)
        paths.append(path)

    # The snapshot folder holds the files at their repo paths (which may include subfolders).
    snapshot = paths[0].parents[len(Path(artifact.files[0].path).parts) - 1] if paths else Path()
    logger.debug("Artifact ready in %s (%d files)", snapshot, len(paths))
    return LocalModel(artifact=artifact, snapshot_dir=snapshot, files=tuple(paths))
