# SPDX-License-Identifier: Apache-2.0
"""scripts/homebrew_formula.py: the Homebrew formula generated from the real uv.lock."""

import importlib.util
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
URL = "https://files.pythonhosted.org/packages/xx/brasskey-0.1.0a1.tar.gz"
SHA = "a" * 64


@pytest.fixture(scope="module")
def generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "homebrew_formula", ROOT / "scripts" / "homebrew_formula.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def formula(generator) -> str:  # type: ignore[no-untyped-def]
    return generator.render(URL, SHA, generator.load_lock())


def resource_names(text: str) -> list[str]:
    return re.findall(r'resource "([^"]+)" do', text)


def test_brasskey_itself_and_homebrew_dependencies(formula):
    assert f'url "{URL}"' in formula
    assert f'sha256 "{SHA}"' in formula
    assert 'depends_on "python@3.15"' in formula
    assert 'depends_on "pydantic" => :no_linkage' in formula
    assert 'depends_on "rust" => :build' in formula
    assert 'license "Apache-2.0"' in formula


def test_resources_are_the_locked_runtime_dependencies(formula, generator):
    names = resource_names(formula)
    lock = generator.load_lock()
    for expected in (
        "hf-xet",
        "huggingface-hub",
        "httpx2",
        "psutil",
        "keyring",
        "uvicorn",
        "typer",
    ):
        assert expected in names, expected
    for version_pin in ("hf-xet", "psutil"):
        sdist = lock[version_pin]["sdist"]
        assert sdist["url"] in formula  # exactly the locked file
        assert sdist["hash"].removeprefix("sha256:") in formula


def test_homebrew_formulas_and_other_platforms_are_not_resources(formula, generator):
    names = set(resource_names(formula))
    from_homebrew = set(generator.FROM_HOMEBREW)
    assert not names & from_homebrew
    for shared in ("pydantic", "pydantic-core", "cryptography", "cffi", "certifi"):
        assert shared in from_homebrew, shared  # provided by homebrew-core, not compiled again
    assert "pywin32-ctypes" not in names  # Windows only
    assert "colorama" not in names  # Windows only
    assert "importlib-metadata" not in names  # Python < 3.12 only
    assert "brasskey" not in names


def test_linux_only_resources_are_in_one_on_linux_block(formula):
    blocks = re.findall(r"\n  on_linux do\n(.*?)\n  end\n", formula, re.DOTALL)
    assert len(blocks) == 1
    assert 'depends_on "cryptography" => :no_linkage' in blocks[0]
    assert set(resource_names(blocks[0])) == {"jeepney", "secretstorage"}


def test_every_resource_is_pinned(formula):
    for match in re.finditer(
        r'resource "([^"]+)" do\n\s+url "([^"]+)"\n\s+sha256 "([^"]+)"', formula
    ):
        name, url, sha = match.groups()
        assert url.startswith("https://files.pythonhosted.org/"), name
        assert url.endswith((".tar.gz", ".zip")), name
        assert re.fullmatch(r"[0-9a-f]{64}", sha), name
    assert len(re.findall(r'resource "', formula)) == len(
        re.findall(r'resource "[^"]+" do\n\s+url "[^"]+"\n\s+sha256 "[0-9a-f]{64}"', formula)
    )


@pytest.mark.skipif(shutil.which("ruby") is None, reason="ruby isn't installed")
def test_ruby_syntax(formula, tmp_path):
    path = tmp_path / "brasskey.rb"
    path.write_text(formula)
    ruby = shutil.which("ruby")
    assert ruby is not None
    result = subprocess.run([ruby, "-c", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_macos_only_dependencies_are_refused(generator):
    lock = {
        "brasskey": {"name": "brasskey", "dependencies": [
            {"name": "appkit-helper", "marker": "sys_platform == 'darwin'"}]},
        "appkit-helper": {"name": "appkit-helper", "dependencies": []},
    }  # fmt: skip
    with pytest.raises(SystemExit, match="on_macos"):
        generator.resources(lock)


def test_sdist_options_go_together(generator):
    with pytest.raises(SystemExit):
        generator.main(["0.1.0a1", "--sdist-url", URL])
