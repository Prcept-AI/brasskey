# SPDX-License-Identifier: Apache-2.0
"""--version, --help and the bare ``brasskey`` command."""

import subprocess
import sys

import pytest

from brasskey import __version__

COMMANDS = ("run", "bench", "doctor")


def test_version_human(cli):
    result = cli("--version")
    assert result.exit_code == 0
    assert f"brasskey {__version__}" in result.stdout
    assert "Python " in result.stdout
    assert result.last_line().startswith("Next: ")
    assert result.stderr == ""


@pytest.mark.parametrize("args", [("--version", "--json"), ("--json", "--version")])
def test_version_json_in_any_order(cli, args):
    result = cli(*args)
    assert result.exit_code == 0
    document = result.json()
    assert document["schema"] == "brasskey.output/v1"
    assert document["command"] == "version"
    assert document["ok"] is True
    assert document["data"]["version"] == __version__
    assert document["next"]["command"]


def test_version_before_a_command_prints_version_and_stops(cli):
    result = cli("--version", "run")
    assert result.exit_code == 0
    assert __version__ in result.stdout
    assert "isn't available" not in result.output


def test_python_dash_m_entry_point():
    process = subprocess.run(
        [sys.executable, "-m", "brasskey", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0
    assert __version__ in process.stdout


def test_no_arguments_shows_help(cli):
    result = cli()
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_flags(cli, flag):
    result = cli(flag)
    assert result.exit_code == 0
    assert "the key to every engine" in result.stdout
    for name in COMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_documents_the_shared_flags(cli, command):
    result = cli(command, "--help")
    assert result.exit_code == 0
    for flag in ("--json", "--debug", "--yes"):
        assert flag in result.stdout


def test_bare_json_lists_commands(cli):
    result = cli("--json")
    assert result.exit_code == 0
    document = result.json()
    assert document["ok"] is True
    assert document["data"]["commands"] == sorted(COMMANDS)
    assert document["data"]["version"] == __version__
