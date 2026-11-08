# SPDX-License-Identifier: Apache-2.0
"""Repository policies: SPDX headers, opt-in test markers and the license checker."""

import importlib.util
import re
import shutil
import subprocess
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


# --- no credentials in the repository -----------------------------------------------------------

# High-signal credential formats. Written so these lines can't match themselves.
CREDENTIAL_PATTERNS = {
    "AWS access key": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b",
    "GitHub token": r"\b(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "PyPI token": r"\bpypi-AgE[A-Za-z0-9_-]{50,}",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "OpenAI / Anthropic key": r"\bsk-(ant-)?[A-Za-z0-9_-]{32,}",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "hard-coded Brasskey key": r"[\"']bk-[A-Za-z0-9_-]{16,}[\"']",
}
SCANNED_SUFFIXES = {".py", ".sh", ".toml", ".yml", ".yaml", ".json", ".md", ".txt", ".cfg"}


def _repository_text_files() -> list[Path]:
    files: list[Path] = []
    for folder in ("src", "tests", "scripts", ".github", "docs", "schemas"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in SCANNED_SUFFIXES)
    files.extend(p for p in ROOT.glob("*") if p.is_file() and p.suffix in SCANNED_SUFFIXES)
    return sorted(files)


def test_no_credentials_in_the_repository():
    """Real tokens never belong in the repo; test keys come from fakes.helpers.fake_api_key()."""
    found = []
    for path in _repository_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in CREDENTIAL_PATTERNS.items():
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{path.relative_to(ROOT)}:{line}: {name}")
    assert not found, "Possible credentials (run scripts/check_secrets.sh):\n" + "\n".join(found)


def test_no_secret_files_are_tracked():
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split()
    risky = [
        name for name in tracked
        if Path(name).name in {".env", ".netrc", ".pypirc", "id_rsa", "id_ed25519"}
        or Path(name).suffix in {".pem", ".key", ".p12", ".pfx"}
        or Path(name).name.startswith(".env.")
    ]  # fmt: skip
    assert not risky, f"Files that usually hold secrets are tracked: {risky}"


# --- container images are pinned by digest -------------------------------------------------------


def test_container_images_are_pinned_by_digest():
    """Every image Brasskey builds from or with names an exact digest, like every CI tool."""
    unpinned = []
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    for image in re.findall(r"^FROM (\S+)", dockerfile, re.MULTILINE):
        if not re.search(r"@sha256:[0-9a-f]{64}$", image):
            unpinned.append(f"docker/Dockerfile: FROM {image}")
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        for name, value in re.findall(r'^\s+(\w+_IMAGE): "([^"]+)"', workflow.read_text(), re.M):
            if not re.search(r"@sha256:[0-9a-f]{64}$", value):
                unpinned.append(f"{workflow.name} {name}={value}")
    assert not unpinned, f"Pin these by digest: {unpinned}"
