# SPDX-License-Identifier: Apache-2.0
"""Error handling in ``brasskey.cli.app.main``: messages, fixes, tracebacks and exit codes.

Commands are made to fail by patching ``apply_flags`` in the command module, which every command
calls first.
"""

import errno

import pytest

from brasskey.errors import BrasskeyError, ExitCode
from fakes import machines

RUN_FLAGS = "brasskey.cli.run.apply_flags"


def _raise(error: BaseException):
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    return fail


def test_unknown_option_is_a_usage_error(cli):
    result = cli("run", "--bogus")
    assert result.exit_code == ExitCode.USAGE
    assert "No such option: --bogus" in result.stderr
    assert "brasskey run --help" in result.stderr
    assert result.last_line().startswith("Next: ")
    assert "Traceback" not in result.output


def test_unknown_command_suggests_the_closest_one(cli):
    result = cli("rnu")
    assert result.exit_code == ExitCode.USAGE
    assert "run" in result.stderr


def test_missing_argument_in_json_mode(cli):
    result = cli("bench", "--json")
    assert result.exit_code == ExitCode.USAGE
    document = result.json()
    assert document["ok"] is False
    assert document["error"]["code"] == "BK-E-USAGE"
    assert "endpoint" in document["error"]["message"].lower()
    assert document["next"]["command"] == "brasskey bench --help"


def test_brasskey_error_uses_its_exit_code_and_fix(cli, monkeypatch):
    error = BrasskeyError(
        "Port 8000 is busy.", fix="Pass --port 8001.", exit_code=ExitCode.CHECK_FAILED
    )
    monkeypatch.setattr(RUN_FLAGS, _raise(error))
    result = cli("run")
    assert result.exit_code == ExitCode.CHECK_FAILED
    assert "Error: Port 8000 is busy." in result.stderr
    assert "Fix: Pass --port 8001." in result.stderr


def test_messages_with_square_brackets_print_literally(cli, monkeypatch):
    monkeypatch.setattr(RUN_FLAGS, _raise(BrasskeyError("Model [qwen3-4b] not found.")))
    result = cli("run")
    assert "[qwen3-4b]" in result.stderr


def test_unexpected_error_hides_traceback_by_default(cli, monkeypatch):
    monkeypatch.setattr(RUN_FLAGS, _raise(RuntimeError("kaboom")))
    result = cli("run")
    assert result.exit_code == ExitCode.ERROR
    assert "unexpected error (RuntimeError: kaboom)" in result.stderr
    assert "Traceback" not in result.output
    assert "Next: brasskey --debug run" in result.stderr


def test_unexpected_error_shows_traceback_with_debug(cli, monkeypatch):
    monkeypatch.setattr(RUN_FLAGS, _raise(RuntimeError("kaboom")))
    result = cli("run", "--debug")
    assert result.exit_code == ExitCode.ERROR
    assert "Traceback" in result.stderr
    assert "kaboom" in result.stderr


def test_unexpected_error_in_json_mode(cli, monkeypatch):
    monkeypatch.setattr(RUN_FLAGS, _raise(RuntimeError("kaboom")))
    result = cli("--json", "run")
    assert result.exit_code == ExitCode.ERROR
    document = result.json()
    assert document["ok"] is False
    assert document["error"]["code"] == "BK-E-INTERNAL"


def test_ctrl_c_exits_130_with_a_short_message(cli, monkeypatch):
    monkeypatch.setattr(RUN_FLAGS, _raise(KeyboardInterrupt()))
    result = cli("run")
    assert result.exit_code == ExitCode.INTERRUPTED
    assert "Stopped (Ctrl-C)." in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "args", [("--debug", "bench", "http://x"), ("bench", "http://x", "--debug")]
)
def test_debug_flag_enables_debug_logs_in_either_position(cli, args):
    result = cli(*args)
    assert "bench requested" in result.stderr


def test_no_debug_logs_by_default(cli):
    result = cli("bench", "http://x")
    assert "bench requested" not in result.stderr


def test_closed_output_pipe_exits_quietly(cli, monkeypatch):
    def broken_pipe(*_args, **_kwargs):
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr("brasskey.cli.doctor.probe_hardware", machines.m3_pro_18gb)
    monkeypatch.setattr("brasskey.cli.app.emit_result", broken_pipe)
    result = cli("doctor", "--json")  # e.g. `brasskey doctor --json | head -c 10`
    assert result.exit_code == ExitCode.BROKEN_PIPE
    assert "Traceback" not in result.output
    assert result.stderr == ""
