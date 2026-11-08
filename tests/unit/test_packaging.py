# SPDX-License-Identifier: Apache-2.0
"""Packaging: README links made absolute for PyPI, and the pre-publish distribution check."""

import importlib.util
import io
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook(monkeypatch_module):  # type: ignore[no-untyped-def]
    """hatch_build.py, with hatchling stubbed (it is only the build backend, not installed)."""
    interface = types.ModuleType("hatchling.metadata.plugin.interface")
    interface.MetadataHookInterface = object  # type: ignore[attr-defined]
    for name in ("hatchling", "hatchling.metadata", "hatchling.metadata.plugin"):
        monkeypatch_module.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch_module.setitem(sys.modules, "hatchling.metadata.plugin.interface", interface)
    return _load("hatch_build", ROOT / "hatch_build.py")


@pytest.fixture(scope="module")
def monkeypatch_module():  # type: ignore[no-untyped-def]
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def check_dist():  # type: ignore[no-untyped-def]
    return _load("check_dist", ROOT / "scripts" / "check_dist.py")


def test_readme_links_become_absolute(hook):  # type: ignore[no-untyped-def]
    text = (
        '<img alt="B" src="brand/logo/a.svg" width="240">\n'
        '<source srcset="brand/logo/b.svg">\n'
        "See [SECURITY.md](SECURITY.md), [decisions](docs/decisions/), "
        "[uv](https://docs.astral.sh/uv/), [top](#status) and ![x](brand/png/x.png)."
    )
    out = hook.absolute_links(text, "v0.1.0a1")
    raw = "https://raw.githubusercontent.com/Prcept-AI/brasskey/v0.1.0a1"
    pages = "https://github.com/Prcept-AI/brasskey"
    assert f'src="{raw}/brand/logo/a.svg"' in out
    assert f'srcset="{raw}/brand/logo/b.svg"' in out
    assert f"[SECURITY.md]({pages}/blob/v0.1.0a1/SECURITY.md)" in out
    assert f"[decisions]({pages}/tree/v0.1.0a1/docs/decisions/)" in out
    assert "[uv](https://docs.astral.sh/uv/)" in out  # already absolute: untouched
    assert "[top](#status)" in out  # anchors: untouched
    assert f"![x]({raw}/brand/png/x.png)" in out


def test_the_real_readme_has_no_relative_links_after_rewriting(hook, check_dist):  # type: ignore[no-untyped-def]
    out = hook.absolute_links((ROOT / "README.md").read_text(), "v0.1.0")
    assert check_dist.RELATIVE_LINK.search(out) is None


def make_wheel(path: Path, *, version: str = "0.1.0a1", description: str = "See [x](https://e/x)",
               license_expr: str = "Apache-2.0", extra: tuple[str, ...] = ()) -> Path:  # fmt: skip
    info = f"brasskey-{version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.4\nName: brasskey\nVersion: {version}\n"
        f"License-Expression: {license_expr}\nDescription-Content-Type: text/markdown\n\n"
        f"{description}\n"
    )
    wheel = path / f"brasskey-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("brasskey/__init__.py", "brasskey/py.typed", "brasskey/models/catalog.yaml",
                     f"{info}/licenses/LICENSE", f"{info}/licenses/NOTICE", *extra):  # fmt: skip
            archive.writestr(name, "x")
        archive.writestr(f"{info}/METADATA", metadata)
    return wheel


def make_sdist(path: Path, *, version: str = "0.1.0a1", tops: tuple[str, ...] = ()) -> Path:
    sdist = path / f"brasskey-{version}.tar.gz"
    base = f"brasskey-{version}"
    names = ("pyproject.toml", "hatch_build.py", "LICENSE", "NOTICE", "PKG-INFO",
             "src/brasskey/__init__.py", *tops)  # fmt: skip
    with tarfile.open(sdist, "w:gz") as archive:
        for name in names:
            data = b"x"
            entry = tarfile.TarInfo(f"{base}/{name}")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return sdist


def test_a_good_distribution_passes(check_dist, tmp_path):  # type: ignore[no-untyped-def]
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    assert check_dist.check(tmp_path, "0.1.0a1") == []
    assert check_dist.main([str(tmp_path), "--version", "0.1.0a1"]) == 0


@pytest.mark.parametrize(
    ("wheel_args", "sdist_args", "expected"),
    [
        ({"description": '<img src="brand/a.svg">'}, {}, "relative link"),
        ({"description": "[S](SECURITY.md)"}, {}, "relative link"),
        ({"license_expr": "MIT"}, {}, "License-Expression is MIT"),
        ({"extra": ("tests/test_x.py",)}, {}, "contains tests/"),
        ({}, {"tops": ("notes.md",)}, "unexpected files: notes.md"),
        ({}, {"version": "0.1.0a2"}, "doesn't match"),
    ],
)
def test_problems_are_caught(check_dist, tmp_path, wheel_args, sdist_args, expected):  # type: ignore[no-untyped-def]
    make_wheel(tmp_path, **wheel_args)
    make_sdist(tmp_path, **sdist_args)
    problems = check_dist.check(tmp_path, None)
    assert any(expected in problem for problem in problems), problems


def test_the_release_tag_must_match_the_version(check_dist, tmp_path):  # type: ignore[no-untyped-def]
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    assert check_dist.check(tmp_path, "0.1.0") == [
        "the version is 0.1.0a1, but the release tag says 0.1.0"
    ]
    assert check_dist.main([str(tmp_path), "--version", "0.1.0"]) == 1


def test_missing_files(check_dist, tmp_path):  # type: ignore[no-untyped-def]
    assert "found 0 and 0" in check_dist.check(tmp_path, None)[0]
