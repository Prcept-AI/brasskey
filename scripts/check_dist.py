# SPDX-License-Identifier: Apache-2.0
"""Check built distributions before they are published (CI and the release workflow run this).

    uv run python scripts/check_dist.py dist [--version 0.1.0a1]

Fails (exit 1) unless:
- there is exactly one wheel and one sdist, with the same version (and ``--version`` if given,
  e.g. the release tag without its ``v``);
- the wheel holds the package data (model catalog, ``py.typed``) and the license files, and no
  tests;
- the metadata says Apache-2.0, and the description (the README) has only absolute links, so it
  renders on PyPI;
- the sdist holds only the published source tree: an allowlist, so nothing else in the working
  folder can ship by accident.
"""

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

WHEEL_MUST_HAVE = ("brasskey/models/catalog.yaml", "brasskey/py.typed", "brasskey/__init__.py")
SDIST_ALLOWED_TOP = frozenset(
    {
        "src", "tests", "scripts", "schemas", "README.md", "LICENSE", "NOTICE", "SECURITY.md",
        "CONTRIBUTING.md", "pyproject.toml", "hatch_build.py", "PKG-INFO", ".gitignore",
    }
)  # fmt: skip
SDIST_MUST_HAVE = ("pyproject.toml", "hatch_build.py", "LICENSE", "NOTICE", "src")
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)[^)\s]+\)|\b(?:src|srcset)=\"(?!https?://)")


def check_wheel(path: Path) -> tuple[str, list[str]]:
    """(version, problems) for a wheel."""
    problems: list[str] = []
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        if metadata_name is None:
            return "", [f"{path.name}: no METADATA"]
        metadata = Parser().parsestr(wheel.read(metadata_name).decode("utf-8"))
    version = metadata.get("Version", "")
    for required in WHEEL_MUST_HAVE:
        if required not in names:
            problems.append(f"{path.name}: missing {required}")
    for license_file in ("LICENSE", "NOTICE"):
        if not any(n.endswith(f".dist-info/licenses/{license_file}") for n in names):
            problems.append(f"{path.name}: missing license file {license_file}")
    if any(n.startswith("tests/") for n in names):
        problems.append(f"{path.name}: contains tests/")
    if metadata.get("License-Expression") != "Apache-2.0":
        problems.append(f"{path.name}: License-Expression is {metadata.get('License-Expression')}")
    if metadata.get("Description-Content-Type") != "text/markdown":
        problems.append(f"{path.name}: description isn't text/markdown")
    description = metadata.get_payload()
    if not isinstance(description, str) or not description.strip():
        problems.append(f"{path.name}: empty description")
    elif match := RELATIVE_LINK.search(description):
        problems.append(f"{path.name}: relative link in the description: {match.group(0)!r}")
    return version, problems


def check_sdist(path: Path) -> list[str]:
    problems: list[str] = []
    with tarfile.open(path) as sdist:
        names = sdist.getnames()
    tops = {name.split("/", 2)[1] for name in names if name.count("/") >= 1}
    unexpected = sorted(tops - SDIST_ALLOWED_TOP)
    if unexpected:
        problems.append(f"{path.name}: unexpected files: {', '.join(unexpected)}")
    for required in SDIST_MUST_HAVE:
        if required not in tops:
            problems.append(f"{path.name}: missing {required}")
    return problems


def check(dist: Path, expected_version: str | None) -> list[str]:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [
            f"expected one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}"
        ]
    version, problems = check_wheel(wheels[0])
    problems += check_sdist(sdists[0])
    if not sdists[0].name.endswith(f"-{version}.tar.gz"):
        problems.append(f"the sdist {sdists[0].name} doesn't match the wheel's version {version}")
    if expected_version is not None and version != expected_version:
        problems.append(f"the version is {version}, but the release tag says {expected_version}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--version", dest="expected_version", default=None)
    args = parser.parse_args(argv)
    problems = check(args.dist, args.expected_version)
    for problem in problems:
        print(f"FAIL  {problem}", file=sys.stderr)
    if not problems:
        print(f"OK    {args.dist}: wheel and sdist are ready to publish")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
