# SPDX-License-Identifier: Apache-2.0
"""Progress display for downloads and verification (stderr; hidden with --json).

Callbacks may arrive from download threads; Rich's Progress is thread-safe.
"""

from collections.abc import Generator
from contextlib import contextmanager

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from brasskey.output.render import stderr_console


class FetchDisplay:
    """One bar for the whole download, one for checksum verification when it runs."""

    def __init__(self, progress: Progress | None) -> None:
        self.progress = progress
        self.download: TaskID | None = None
        self.verify: TaskID | None = None

    def on_download(self, done: int, total: int) -> None:
        if self.progress is None:
            return
        if self.download is None:
            self.download = self.progress.add_task("Downloading", total=total)
        self.progress.update(self.download, completed=done, total=total)

    def on_verify(self, done: int, total: int) -> None:
        if self.progress is None:
            return
        if self.verify is None:
            self.verify = self.progress.add_task("Verifying checksums", total=total)
        self.progress.update(self.verify, completed=done, total=total)


@contextmanager
def fetch_display(*, enabled: bool) -> Generator[FetchDisplay, None, None]:
    if not enabled:
        yield FetchDisplay(None)
        return
    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=stderr_console(), transient=True) as progress:
        yield FetchDisplay(progress)
