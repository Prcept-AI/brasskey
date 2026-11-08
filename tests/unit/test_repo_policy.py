# SPDX-License-Identifier: Apache-2.0
"""Repository policies: SPDX headers, opt-in test markers and the license checker."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPDX_LINE = "# SPDX-License-Identifier: Apache-2.0"


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for folder in ("src", "tests", "scripts"):
        files.extend(sorted((ROOT / folder).rglob("*.py")))
    return files


def test_python_sources_exist():
    assert any(path.name == "app.py" for path in _python_sources())


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_spdx_header(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    assert lines, f"{path} is empty; add the SPDX header"
    assert lines[0] == SPDX_LINE, f"{path} must start with '{SPDX_LINE}'"


# --- opt-in markers (gpu / engine / network) ----------------------------------------------------

SAMPLE_TESTS = """
import pytest

@pytest.mark.gpu
def test_needs_a_gpu():
    pass

def test_plain():
    pass
"""

MARKER_INI = """
[pytest]
markers =
    gpu: x
    engine: x
    network: x
"""


def test_opt_in_markers_are_skipped_by_default(pytester):
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini(MARKER_INI)
    pytester.makepyfile(test_sample=SAMPLE_TESTS)
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)


def test_opt_in_markers_run_when_selected(pytester):
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini(MARKER_INI)
    pytester.makepyfile(test_sample=SAMPLE_TESTS)
    pytester.runpytest("-m", "gpu").assert_outcomes(passed=1, deselected=1)


# --- license checker ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def licenses() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_licenses", ROOT / "scripts" / "check_licenses.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("expression", "status"),
    [
        ("MIT", "allowed"),
        ("Apache-2.0", "allowed"),
        ("BSD-3-Clause", "allowed"),
        ("MPL-2.0", "allowed"),
        ("PSF-2.0", "allowed"),
        ("Apache-2.0 OR BSD-2-Clause", "allowed"),
        ("Apache-2.0 WITH LLVM-exception", "allowed"),
        ("MIT OR GPL-2.0-only", "allowed"),
        ("GPL-3.0-only", "denied"),
        ("LGPL-2.1-or-later", "denied"),
        ("AGPL-3.0-only", "denied"),
        ("GPL-2.0-only WITH Classpath-exception-2.0", "denied"),
        ("MIT AND GPL-3.0-only", "denied"),
        ("(MIT OR Apache-2.0) AND GPL-3.0-only", "denied"),
        ("(MIT OR GPL-3.0-only) AND BSD-3-Clause", "allowed"),
        ("SSPL-1.0", "denied"),
        ("LicenseRef-Proprietary", "unknown"),
    ],
)
def test_classify_expression(licenses, expression, status):
    assert licenses.classify_expression(expression) == status


def test_classifiers_and_free_form_fields(licenses):
    check = licenses.check_distribution
    bsd = check(
        "a",
        "1",
        expression=None,
        classifiers=["License :: OSI Approved :: BSD License"],
        license_field=None,
    )
    assert bsd.status == "allowed"
    dual = check(
        "b",
        "1",
        expression=None,
        license_field=None,
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
    )
    assert dual.status == "unknown"
    gpl_text = check("c", "1", expression=None, classifiers=[], license_field="GNU GPL v3")
    assert gpl_text.status == "denied"
    nothing = check("d", "1", expression=None, classifiers=[], license_field="UNKNOWN")
    assert nothing.status == "unknown"
