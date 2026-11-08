# SPDX-License-Identifier: Apache-2.0
"""brasskey.yaml / brasskey.lock: validation, round trips, privacy and schema drift."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from brasskey.config.intent import GoalIntent, Intent, ServeIntent, is_loopback
from brasskey.config.io import (
    ConfigFileError,
    intent_sha256,
    read_intent,
    read_lock,
    write_intent,
    write_lock,
)
from brasskey.config.schema import render_schemas
from brasskey.engines.types import EngineId
from brasskey.models.catalog import load_catalog
from brasskey.planner import RunRequest, plan_run
from brasskey.planner.replay import lock_from_plan
from fakes import machines

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def test_intent_defaults():
    intent = Intent()
    assert intent.model is None
    assert intent.goal.context_length == 8192
    assert intent.serve.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_loopback_hosts_are_allowed(host):
    assert is_loopback(host)
    assert ServeIntent(host=host).host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])  # noqa: S104
def test_other_hosts_are_valid_intent_the_key_rule_applies_at_run_time(host):
    assert not is_loopback(host)
    assert ServeIntent(host=host).host == host


def test_tls_intent():
    serve = ServeIntent.model_validate({"tls": {"cert_file": "c.pem", "key_file": "k.pem"}})
    assert serve.tls is not None
    assert str(serve.tls.cert_file) == "c.pem"


def test_api_key_env_must_be_a_variable_name_not_a_key():
    assert ServeIntent(api_key_env="BRASSKEY_API_KEY").api_key_env == "BRASSKEY_API_KEY"
    with pytest.raises(ValidationError):
        ServeIntent(api_key_env="sk-live-1234")


def test_intent_round_trip_and_stable_hash(tmp_path):
    intent = Intent(model="qwen3-4b", goal=GoalIntent(context_length=16384))
    write_intent(tmp_path, intent)
    text = (tmp_path / "brasskey.yaml").read_text()
    assert text.startswith("# Brasskey intent")
    loaded = read_intent(tmp_path)
    assert loaded == intent
    assert loaded is not None
    assert intent_sha256(intent) == intent_sha256(loaded)


def test_unknown_fields_and_bad_yaml_are_plain_errors(tmp_path):
    (tmp_path / "brasskey.yaml").write_text("version: 1\nmodle: qwen3-4b\n")
    with pytest.raises(ConfigFileError) as caught:
        read_intent(tmp_path)
    assert "modle" in caught.value.message
    (tmp_path / "brasskey.yaml").write_text("version: [1\n")
    with pytest.raises(ConfigFileError):
        read_intent(tmp_path)


def test_missing_files_are_none(tmp_path):
    assert read_intent(tmp_path) is None
    assert read_lock(tmp_path) is None


def test_lock_round_trip_keeps_no_machine_paths(tmp_path):
    hw = machines.m3_pro_18gb()
    plan = plan_run(RunRequest(model="qwen3-4b", engine=EngineId.LLAMACPP), hw, load_catalog())
    model_file = "/Users/someone/.cache/huggingface/hub/x/Qwen3-4B-Q4_K_M.gguf"
    lock = lock_from_plan(
        plan,
        hw,
        intent=Intent(model="qwen3-4b"),
        host="127.0.0.1",
        port=None,
        launch_args=("/opt/homebrew/bin/llama-server", "--model", model_file, "--port", "8000"),
        model_file=model_file,
        startup_seconds=3.456,
    )
    write_lock(tmp_path, lock)
    text = (tmp_path / "brasskey.lock").read_text()
    assert "/Users/someone" not in text
    assert "/opt/homebrew" not in text
    assert "{model}" in text
    loaded = read_lock(tmp_path)
    assert loaded is not None
    assert loaded == lock
    assert loaded.measured.started is True
    assert loaded.measured.startup_seconds == 3.46
    assert loaded.artifact.revision == plan.fit.artifact.revision


@pytest.mark.parametrize("name", sorted(render_schemas()))
def test_committed_json_schemas_match_the_models(name):
    committed = (SCHEMAS_DIR / name).read_text()
    assert committed == render_schemas()[name], (
        "Schemas are stale: run `uv run python scripts/export_schemas.py`"
    )
