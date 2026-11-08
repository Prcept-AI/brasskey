# SPDX-License-Identifier: Apache-2.0
"""scripts/release_notes.py: the CHANGELOG.md section that becomes the GitHub release notes."""

import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CHANGELOG = """# Changelog

## [Unreleased]

### Fixed

- Something not released yet.

## [0.2.0] - 2026-12-01

### Added

- A feature.

## [0.1.0] - 2026-11-08

- The first release.

[Unreleased]: https://github.com/Prcept-AI/brasskey/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Prcept-AI/brasskey/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Prcept-AI/brasskey/releases/tag/v0.1.0
"""


@pytest.fixture(scope="module")
def notes() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / "scripts" / "release_notes.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_section_ends_at_the_next_version(notes):
    assert notes.section(CHANGELOG, "0.2.0") == "### Added\n\n- A feature."


def test_the_last_section_leaves_out_the_link_references(notes):
    assert notes.section(CHANGELOG, "0.1.0") == "- The first release."


def test_unknown_versions_and_empty_sections_fail(notes, tmp_path, capsys):
    assert notes.section(CHANGELOG, "9.9.9") is None
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG + "\n## [0.3.0] - 2027-01-01\n")
    assert notes.main(["0.3.0", "--changelog", str(path)]) == 1  # heading without notes
    assert notes.main(["9.9.9", "--changelog", str(path)]) == 1
    assert "no notes for 9.9.9" in capsys.readouterr().err


def test_every_released_version_has_notes_in_the_real_changelog(notes, capsys):
    tags = ("0.1.0a1", "0.1.0a2")
    for version in tags:
        assert notes.main([version]) == 0, version
    printed = capsys.readouterr().out
    assert "Homebrew" in printed  # 0.1.0a2
    assert "The first release." in printed  # 0.1.0a1
    assert "https://github.com/Prcept-AI/brasskey/compare" not in printed
