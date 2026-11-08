# SPDX-License-Identifier: Apache-2.0
"""A fake command runner that replays sysctl and ioreg the way the real tools behave."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from brasskey.hardware.commands import IOREG, SYSCTL, CommandOutput
from fakes import HARDWARE_FIXTURES


def load_sysctl(name: str) -> dict[str, str]:
    """Load a captured ``sysctl`` fixture ("name: value" lines)."""
    values: dict[str, str] = {}
    for line in (HARDWARE_FIXTURES / name).read_text().splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            values[key.strip()] = value.strip()
    return values


def load_text(name: str) -> str:
    return (HARDWARE_FIXTURES / name).read_text()


class FakeRunner:
    """Answers like sysctl (known names and subtrees, errors for unknown ones) and ioreg."""

    def __init__(self, sysctl: Mapping[str, str] | None = None, ioreg: str = "") -> None:
        self.sysctl = dict(sysctl or {})
        self.ioreg = ioreg
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], *, timeout_s: float = 5.0) -> CommandOutput:
        self.calls.append(tuple(argv))
        if argv[0] == SYSCTL:
            lines: list[str] = []
            unknown: list[str] = []
            for name in argv[1:]:
                keys = [key for key in self.sysctl if key == name or key.startswith(name + ".")]
                if not keys:
                    unknown.append(name)
                lines.extend(f"{key}: {self.sysctl[key]}" for key in keys)
            stderr = "".join(f"sysctl: unknown oid '{name}'\n" for name in unknown)
            return CommandOutput(1 if unknown else 0, "\n".join(lines) + "\n", stderr)
        if argv[0] == IOREG:
            return CommandOutput(0, self.ioreg)
        return CommandOutput(127, "", f"{argv[0]} not found")


def write_tree(root: Path, files: Mapping[str, str]) -> Path:
    """Create files under ``root`` (e.g. a fake /proc or /sys) and return ``root``."""
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root
