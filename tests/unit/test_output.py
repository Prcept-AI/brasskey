# SPDX-License-Identifier: Apache-2.0
"""Output building blocks: settings prescan, JSON envelope, Next step, confirmation, logging."""

import json
import logging

import pytest
from pydantic import ValidationError

from brasskey.errors import ConfirmationRequiredError
from brasskey.log import LOGGER_NAME, _HandlerSlot, configure_logging
from brasskey.output.confirm import confirm
from brasskey.output.envelope import Envelope, ErrorInfo
from brasskey.output.next_step import NextStep
from brasskey.output.settings import OutputSettings

# --- settings prescan ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], OutputSettings()),
        (["--json", "run"], OutputSettings(json=True, command="run")),
        (["run", "--debug", "-y"], OutputSettings(debug=True, yes=True, command="run")),
        (["--yes", "bench", "http://x"], OutputSettings(yes=True, command="bench")),
        # Everything after "--" belongs to the command, not to Brasskey.
        (["bench", "http://x", "--", "--json"], OutputSettings(command="bench")),
    ],
)
def test_prescan(argv, expected):
    assert OutputSettings.prescan(argv) == expected


# --- envelope and next step ---------------------------------------------------------------------


def test_envelope_field_names_and_order():
    envelope = Envelope(command="run", ok=True, data={"a": 1}, next=NextStep(command="brasskey x"))
    document = json.loads(envelope.to_json())
    assert list(document) == ["schema", "command", "ok", "data", "error", "next"]
    assert document["schema"] == "brasskey.output/v1"
    assert document["next"] == {"command": "brasskey x", "hint": None}


def test_error_info_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ErrorInfo.model_validate({"code": "X", "message": "m", "surprise": True})


@pytest.mark.parametrize(
    ("step", "text"),
    [
        (NextStep(command="brasskey run"), "brasskey run"),
        (NextStep(command="brasskey run", hint="start a model"), "brasskey run (start a model)"),
    ],
)
def test_next_step_text(step, text):
    assert step.as_text() == text


def test_next_step_requires_a_command():
    with pytest.raises(ValidationError):
        NextStep(command="")


# --- confirmation -------------------------------------------------------------------------------


def _never_called() -> bool:
    pytest.fail("the prompt should not be consulted")


def test_yes_skips_the_prompt():
    assert confirm(
        "Download 2.6 GB?", settings=OutputSettings(yes=True), is_interactive=_never_called
    )


def test_json_mode_needs_yes():
    with pytest.raises(ConfirmationRequiredError) as caught:
        confirm("Download 2.6 GB?", settings=OutputSettings(json=True), is_interactive=lambda: True)
    assert "--yes" in (caught.value.fix or "")
    assert caught.value.code == "BK-E-CONFIRM"


def test_non_interactive_stdin_needs_yes():
    with pytest.raises(ConfirmationRequiredError):
        confirm("Download 2.6 GB?", settings=OutputSettings(), is_interactive=lambda: False)


@pytest.mark.parametrize("answer", [True, False])
def test_interactive_prompt_returns_the_answer(monkeypatch, answer):
    monkeypatch.setattr("brasskey.output.confirm.Confirm.ask", lambda *_a, **_k: answer)
    assert confirm("Go?", settings=OutputSettings(), is_interactive=lambda: True) is answer


# --- logging ------------------------------------------------------------------------------------


def test_configure_logging_replaces_its_handler_and_sets_level():
    # Only our own handler is checked: pytest attaches capture handlers to loggers too.
    logger = logging.getLogger(LOGGER_NAME)
    configure_logging(debug=False, json_mode=False)
    first = _HandlerSlot.handler
    configure_logging(debug=True, json_mode=True)
    second = _HandlerSlot.handler
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert first not in logger.handlers
    assert second in logger.handlers
    configure_logging(debug=False, json_mode=False)
    assert logger.level == logging.WARNING
    assert second not in logger.handlers
