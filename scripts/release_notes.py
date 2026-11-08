# SPDX-License-Identifier: Apache-2.0
"""Print the CHANGELOG.md section for a version, for the GitHub release notes.

    python3 scripts/release_notes.py 0.1.0a3 [--changelog CHANGELOG.md]

The release workflow runs this for every tag and fails (exit 1) when the version has no section,
or an empty one, so a release can't ship without its notes (RELEASING.md, step 1).
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def section(text: str, version: str) -> str | None:
    """The body under ``## [version]`` up to the next ``## [`` heading (None: no such section)."""
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is None:
            continue
        if start is not None:  # the next version's heading ends the section
            return "\n".join(lines[start:index]).strip()
        if match["version"] == version:
            start = index + 1
    if start is None:
        return None
    # The last section ends before the link references ("[0.1.0a1]: https://...").
    body = [line for line in lines[start:] if not re.match(r"^\[[^\]]+\]: ", line)]
    return "\n".join(body).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("version", help="The release version, e.g. 0.1.0a3 (no leading v).")
    parser.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    args = parser.parse_args(argv)
    version: str = args.version
    changelog: Path = args.changelog
    notes = section(changelog.read_text(encoding="utf-8"), version)
    if not notes:
        print(
            f"release_notes: {changelog.name} has no notes for {version}. Move the Unreleased "
            f"entries under '## [{version}] - <date>' (RELEASING.md, step 1).",
            file=sys.stderr,
        )
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
