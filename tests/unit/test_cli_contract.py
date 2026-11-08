# SPDX-License-Identifier: Apache-2.0
"""The CLI contract every command must keep (CONTRIBUTING.md, CLI contract).

- the output ends with a one-line ``Next:`` suggestion;
- ``--json`` works before or after the command and prints exactly one JSON document to stdout;
- no raw tracebacks by default.

SAMPLE_ARGS must list every registered command, so new commands are held to these rules
automatically. Update a command's sample arguments when it gains real behavior (use fakes).
"""

import pytest

from brasskey.cli.app import app

SAMPLE_ARGS: dict[str, list[str]] = {
    "run": ["--dry-run", "--offline"],
    "bench": ["http://127.0.0.1:8000/v1"],
    "doctor": [],
}


def _registered_commands() -> list[str]:
    return sorted(info.name for info in app.registered_commands if info.name)


def test_every_registered_command_has_sample_args():
    assert sorted(SAMPLE_ARGS) == _registered_commands()


@pytest.mark.parametrize("command", sorted(SAMPLE_ARGS))
def test_human_output_ends_with_next(cli, command):
    result = cli(command, *SAMPLE_ARGS[command])
    assert result.last_line().startswith("Next: ")
    assert "Traceback" not in result.output


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("command", sorted(SAMPLE_ARGS))
def test_json_is_one_document_with_next(cli, command, position):
    sample = SAMPLE_ARGS[command]
    args = ["--json", command, *sample] if position == "before" else [command, *sample, "--json"]
    result = cli(*args)
    document = result.json()  # fails unless stdout is exactly one JSON document
    assert document["schema"] == "brasskey.output/v1"
    assert document["command"] == command
    assert document["next"] is not None
    assert document["next"]["command"]
    # In --json mode nothing else may be written; logs (none by default) would go to stderr.
    assert result.stderr == ""


@pytest.mark.parametrize("extra", [[], ["http://gpu-box:8000/v1"]])
def test_doctor_security_keeps_the_contract(cli, extra):
    """Exit code 3 (a check failed) still prints one JSON document with ok=true and Next."""
    document = cli("doctor", "--security", *extra, "--json").json()
    assert document["ok"] is True
    assert document["command"] == "doctor"
    assert isinstance(document["data"]["checks"], list)
    assert document["next"]["command"].startswith("brasskey ")
