# SPDX-License-Identifier: Apache-2.0
"""Shared test setup.

- Opt-in markers (``gpu``, ``engine``, ``network``) are skipped unless selected with ``-m``, so
  ``uv run pytest`` passes on a laptop without a GPU and ``uv run pytest -m gpu`` runs them.
- The ``cli`` fixture runs the real ``brasskey`` entry point in-process and captures its output.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest

from brasskey.cli.app import main
from brasskey.security.doctor import SecurityContext
from brasskey.security.netinfo import HttpProbe

pytest_plugins = ("pytester",)

OPT_IN_MARKERS = ("gpu", "engine", "network")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in tests unless the -m expression names their marker."""
    markexpr = str(config.getoption("markexpr") or "")
    for item in items:
        for name in OPT_IN_MARKERS:
            if item.get_closest_marker(name) is not None and name not in markexpr:
                item.add_marker(pytest.mark.skip(reason=f"opt-in: run with `pytest -m {name}`"))


@pytest.fixture(autouse=True)
def _private_brasskey_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never touch the developer's real Brasskey folders (state files, logs) from tests."""
    monkeypatch.setenv("BRASSKEY_HOME", str(tmp_path_factory.mktemp("brasskey-home")))


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the developer's API key or touch the real OS keychain."""
    monkeypatch.delenv("BRASSKEY_API_KEY", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setattr("brasskey.cli.run.keyring_reader", lambda _service, _user: None)
    monkeypatch.setattr("brasskey.cli.run.keyring_writer", _refuse_keyring)
    monkeypatch.setattr("brasskey.cli.bench.keyring_reader", lambda _service, _user: None)


def _refuse_keyring(_service: str, _user: str, _value: str) -> None:
    raise RuntimeError("tests must not write to the OS keychain")


@pytest.fixture(autouse=True)
def _no_real_network_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """``doctor --security`` in unit tests sees a quiet fake machine, never this one.

    The real scan runs lsof, ``docker ps`` and HTTP probes of local ports; tests that need a
    particular machine state patch ``gather_security_context`` themselves.
    """

    def closed(url: str) -> HttpProbe:
        return HttpProbe(url=url, status=401, allow_origin=None, looks_like_models=False)

    quiet = SecurityContext(
        servers=[], listeners=[], addresses=[], docker=[], firewall="on", processes=[],
        api_key=None, probe=closed, os_family="linux",
    )  # fmt: skip
    monkeypatch.setattr("brasskey.cli.doctor.gather_security_context", lambda _family: quiet)
    monkeypatch.setattr("brasskey.cli.doctor.make_url_prober", lambda: closed)


@pytest.fixture(autouse=True)
def _no_real_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`brasskey bench` in unit tests reaches no endpoint, GuideLLM or Hugging Face.

    Nothing answers HTTP requests and GuideLLM is "not installed"; tests that need more patch
    the same seams (see tests/unit/test_cli_bench.py).
    """

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("unit tests make no network calls", request=request)

    monkeypatch.setattr(
        "brasskey.cli.bench.make_http_client",
        lambda: httpx2.Client(transport=httpx2.MockTransport(refuse)),
    )
    monkeypatch.setattr("brasskey.cli.bench.detect_guidellm_install", lambda: None)


@pytest.fixture(autouse=True)
def _stable_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic, colorless output regardless of the developer's terminal or CI runner."""
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def json(self) -> dict[str, Any]:
        """Parse stdout as exactly one JSON document (fails if there is anything else)."""
        document = json.loads(self.stdout)
        assert isinstance(document, dict)
        return document

    def last_line(self) -> str:
        lines = [line for line in self.output.splitlines() if line.strip()]
        return lines[-1] if lines else ""


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Callable[..., CliResult]:
    """Run ``brasskey <args>`` in-process; returns exit code, stdout and stderr."""

    def invoke(*args: str) -> CliResult:
        exit_code = main(list(args))
        captured = capsys.readouterr()
        return CliResult(exit_code, captured.out, captured.err)

    return invoke
