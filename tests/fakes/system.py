# SPDX-License-Identifier: Apache-2.0
"""Fake system counters for the CPU probe (instead of psutil)."""

from dataclasses import dataclass

from brasskey.hardware.types import GIB


@dataclass
class FakeStats:
    physical: int | None = 8
    logical: int | None = 16
    total: int = 64 * GIB
    available: int = 48 * GIB
    usable: int | None = None

    def cpu_count(self, *, logical: bool) -> int | None:
        return self.logical if logical else self.physical

    def memory(self) -> tuple[int, int]:
        return self.total, self.available

    def usable_cpus(self) -> int | None:
        return self.usable
