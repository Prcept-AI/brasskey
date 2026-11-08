# SPDX-License-Identifier: Apache-2.0
"""``brasskey run`` end to end: dry runs, and real runs that start the fake engine process."""

import os

import httpx2
import pytest

from brasskey.engines.types import EngineId
from brasskey.paths import resolve_paths
from brasskey.runtime.state import list_records
from fakes import machines
from fakes.engine import (
    FAKE_INSTALL,
    FAKE_MLX_INSTALL,
    FAKE_VLLM_INSTALL,
    TINY_CATALOG,
    TINY_CONTENT,
    FakeAdapter,
    FakeMlxAdapter,
    FakeVllmAdapter,
)
from fakes.hub import FakeHub


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Fake hardware, engine detection, Hugging Face, speed test; a scratch working folder."""
    state = {
        "hw": machines.m3_pro_18gb(),
        "installed": True,
        "mlx": None,  # MLX install to report (None: not installed)
        "hub": FakeHub(cache_dir=tmp_path / "hf-cache", content=TINY_CONTENT),
        "adapter": FakeAdapter(),
        "probes": [],
    }

    def bandwidth(url):
        state["probes"].append(url)
        return 50e6

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("brasskey.cli.run.probe_hardware", lambda: state["hw"])
    monkeypatch.setattr(
        "brasskey.cli.run.detect_engines",
        lambda: {
            EngineId.LLAMACPP: FAKE_INSTALL if state["installed"] else None,
            EngineId.MLX: state["mlx"],
        },
    )
    monkeypatch.setattr("brasskey.cli.run.make_hub", lambda *, offline: state["hub"])
    monkeypatch.setattr("brasskey.cli.run.measure_bandwidth", bandwidth)
    monkeypatch.setattr("brasskey.cli.run.adapter_for", lambda _engine: state["adapter"])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return state


@pytest.fixture
def tiny(world, monkeypatch):
    """Use the tiny test catalog, and stand in for the user while the server runs."""
    monkeypatch.setattr("brasskey.cli.run.load_catalog", lambda: TINY_CATALOG)

    def serve(handle, supervisor, wait=None):
        """Talk to the server through the proxy, as a client would, then stop it."""
        records = list_records(resolve_paths())
        base = records[0].base_url
        key = os.environ.get("BRASSKEY_API_KEY")
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": "tiny", "messages": [{"role": "user", "content": "secret prompt"}]}
        with httpx2.Client(timeout=5, trust_env=False) as http:
            world["models"] = http.get(f"{base}/models", headers=auth).json()
            world["chat"] = http.post(f"{base}/chat/completions", headers=auth, json=body).json()
            world["no_key_status"] = http.get(f"{base}/models").status_code
            world["direct_engine_status"] = http.post(
                f"http://127.0.0.1:{records[0].engine_port}/v1/chat/completions", json=body
            ).status_code
        world["records"] = records
        world["handle"] = handle
        supervisor.stop(handle)
        return None

    monkeypatch.setattr("brasskey.cli.run.serve_until_stopped", serve)
    return world


# --- dry runs (real catalog) ----------------------------------------------------------------------


def test_dry_run_picks_an_installed_engine(cli, world):
    result = cli("run", "--dry-run")
    assert result.exit_code == 0
    assert "Qwen3 4B (qwen3-4b)" in result.stdout
    assert "llama.cpp 0.0.0-fake" in result.stdout
    assert "at 50 MB/s (measured)" in result.stdout
    assert result.last_line() == "Next: brasskey run qwen3-4b (start it)"


def test_dry_run_when_no_engine_is_installed_points_to_the_install(cli, world):
    world["installed"] = False
    result = cli("run", "--dry-run")
    assert "MLX (not installed: brew install fake-engine)" in result.stdout  # preferred on a Mac
    assert result.last_line() == "Next: brew install fake-engine (install MLX)"


def test_dry_run_json(cli, world):
    data = cli("run", "qwen3-4b", "--dry-run", "--json").json()["data"]
    assert data["dry_run"] is True
    assert data["plan_source"] == "command line"
    assert data["engine_installed"] is True
    assert data["plan"]["engine"] == "llamacpp"
    assert "artifacts" not in data["plan"]["model"]
    assert data["plan"]["install"]["version"] == "0.0.0-fake"


def test_offline_skips_the_speed_test(cli, world):
    assert "speed not measured: offline" in cli("run", "--dry-run", "--offline").stdout
    assert world["probes"] == []


def test_mlx_is_preferred_on_a_mac_when_available(cli, world):
    world["mlx"] = FAKE_MLX_INSTALL
    result = cli("run", "--dry-run")
    assert "Qwen3 8B (qwen3-8b)" in result.stdout  # MLX's 4-bit 8B fits the quick-start download
    assert "MLX 0.0.0-fake-mlx" in result.stdout


def test_mlx_setup_is_announced_in_the_dry_run(cli, world):
    world["mlx"] = FAKE_MLX_INSTALL.model_copy(
        update={"needs_setup": True, "setup_note": "about 70 MB, one time"}
    )
    result = cli("run", "--dry-run")
    assert "set up automatically on first run: about 70 MB, one time" in result.stdout


def test_vllm_is_not_startable_on_a_mac(cli, world):
    result = cli("run", "qwen3-4b", "--dry-run", "--engine", "vllm")
    assert result.exit_code == 1
    assert "can't run on this machine" in result.stderr


def test_dry_run_save_writes_both_files(cli, world, tmp_path):
    result = cli("run", "qwen3-4b", "--dry-run", "--save", "--offline")
    assert result.exit_code == 0
    assert (tmp_path / "brasskey.yaml").is_file()
    assert "started: false" in (tmp_path / "brasskey.lock").read_text()


def test_model_that_does_not_fit_explains_and_suggests(cli, world):
    result = cli("run", "qwen3-32b", "--dry-run")
    assert result.exit_code == 1
    assert "doesn't fit here" in result.stderr
    assert "Next: brasskey run qwen3-14b --dry-run" in result.stderr


# --- real runs with the fake engine ---------------------------------------------------------------


def test_run_downloads_starts_serves_and_stops(cli, tiny):
    result = cli("run", "tiny", "--yes")
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Tiny Test is running." in out
    assert "OpenAI-compatible" in out
    assert "curl http://127.0.0.1:" in out
    assert "Next: curl http://127.0.0.1:" in out
    assert "Stopped Tiny Test (llama.cpp)." in result.stderr
    assert tiny["models"]["data"][0]["id"] == "tiny"
    assert tiny["chat"]["choices"][0]["message"]["content"]
    assert [record.model_id for record in tiny["records"]] == ["tiny"]  # visible while running
    assert tiny["records"][0].engine_port != tiny["records"][0].port  # engine is private
    assert tiny["direct_engine_status"] == 401  # nobody can bypass the proxy to llama.cpp
    assert "no API key (this machine only)" in out
    assert list_records(resolve_paths()) == []  # removed on stop
    assert tiny["handle"].process.poll() is not None  # the engine is gone
    assert tiny["hub"].downloads == ["tiny-Q4_K_M.gguf"]
    assert "secret prompt" not in tiny["handle"].log_path.read_text()  # prompts never logged


def test_run_json_prints_one_ready_document(cli, tiny):
    result = cli("run", "tiny", "--yes", "--json")
    assert result.exit_code == 0
    document = result.json()
    assert document["data"]["status"] == "ready"
    assert document["data"]["base_url"].startswith("http://127.0.0.1:")
    assert document["data"]["startup_seconds"] > 0
    assert "Stopped" not in result.stderr


def test_save_then_replay_the_lock(cli, tiny, tmp_path):
    assert cli("run", "tiny", "--yes", "--save").exit_code == 0
    lock = (tmp_path / "brasskey.lock").read_text()
    assert "started: true" in lock  # ({model} placeholders: see test_config.py, real adapter)
    assert str(tmp_path) not in lock
    replay = cli("run", "--yes", "--json").json()["data"]
    assert replay["plan_source"] == "brasskey.lock"
    assert tiny["hub"].downloads == ["tiny-Q4_K_M.gguf"]  # the second run used the cache


def test_brasskey_yaml_is_used_and_locked(cli, tiny, tmp_path):
    (tmp_path / "brasskey.yaml").write_text(
        "version: 1\nmodel: tiny\ngoal:\n  context_length: 1024\n"
    )
    data = cli("run", "--yes", "--json").json()["data"]
    assert data["plan_source"] == "brasskey.yaml"
    assert (tmp_path / "brasskey.lock").is_file()


def test_not_installed_stops_before_downloading(cli, tiny):
    tiny["installed"] = False
    result = cli("run", "tiny", "--yes")
    assert result.exit_code == 1
    assert "isn't installed" in result.stderr
    assert "Next: brew install fake-engine" in result.stderr
    assert tiny["hub"].downloads == []


def test_without_yes_and_no_terminal_nothing_happens(cli, tiny):
    result = cli("run", "tiny")
    assert result.exit_code == 1
    assert "--yes" in result.stderr
    assert tiny["hub"].downloads == []


def test_engine_crash_is_explained_and_cleaned_up(cli, tiny):
    tiny["adapter"] = FakeAdapter("crash")
    result = cli("run", "tiny", "--yes")
    assert result.exit_code == 1
    assert "ran out of memory" in result.stderr
    assert list_records(resolve_paths()) == []


def test_doctor_lists_engines(cli, monkeypatch):
    monkeypatch.setattr("brasskey.cli.doctor.probe_hardware", machines.m3_pro_18gb)
    monkeypatch.setattr(
        "brasskey.cli.doctor.detect_engines", lambda: {EngineId.LLAMACPP: FAKE_INSTALL}
    )
    result = cli("doctor")
    assert "Engines" in result.stdout
    assert "llama.cpp 0.0.0-fake" in result.stdout
    assert result.last_line().startswith("Next: brasskey run --dry-run")
    monkeypatch.setattr("brasskey.cli.doctor.detect_engines", lambda: {EngineId.LLAMACPP: None})
    missing = cli("doctor", "--json").json()
    assert missing["data"]["engines"] == {"llamacpp": None}
    assert missing["next"]["command"] == "brew install llama.cpp"


def test_real_run_with_mlx_runs_the_one_time_setup_first(cli, tiny, monkeypatch):
    tiny["mlx"] = FAKE_MLX_INSTALL.model_copy(
        update={
            "needs_setup": True,
            "setup_note": "about 70 MB, one time",
            "setup_argv": ("uv", "tool", "run"),
        }
    )
    tiny["adapter"] = FakeMlxAdapter()
    setups: list[str] = []
    monkeypatch.setattr(
        "brasskey.cli.run.run_engine_setup", lambda install: setups.append(install.version)
    )
    result = cli("run", "tiny", "--engine", "mlx", "--yes")
    assert result.exit_code == 0, result.output
    assert setups == ["0.0.0-fake-mlx"]
    assert "Setup: MLX (about 70 MB, one time)" in result.stderr
    assert "Stopped Tiny Test (MLX)." in result.stderr


STRONG_KEY = "bk-7Hc2Lq9XzP4mVw8RtY6nJ3aK5sD1fG0e"
ANY_HOST = "0.0.0.0"  # noqa: S104 (tests of the network-exposure rules)


def test_mlx_gets_api_keys_through_the_proxy(cli, tiny, monkeypatch):
    monkeypatch.setenv("BRASSKEY_API_KEY", STRONG_KEY)
    tiny["mlx"] = FAKE_MLX_INSTALL
    tiny["adapter"] = FakeMlxAdapter()
    result = cli("run", "tiny", "--engine", "mlx", "--yes")
    assert result.exit_code == 0, result.output
    assert tiny["no_key_status"] == 401
    assert tiny["models"]["data"][0]["id"] == "tiny"


def test_a_key_from_the_environment_is_enforced_and_never_written(cli, tiny, tmp_path, monkeypatch):
    monkeypatch.setenv("BRASSKEY_API_KEY", STRONG_KEY)
    result = cli("run", "tiny", "--yes", "--save")
    assert result.exit_code == 0, result.output
    assert "API key required (from BRASSKEY_API_KEY)" in result.stdout
    assert "Authorization: Bearer $BRASSKEY_API_KEY" in result.stdout
    assert tiny["no_key_status"] == 401
    assert tiny["chat"]["choices"]
    for path in [*tmp_path.rglob("*"), *list((tmp_path.parent).glob("brasskey-home*/**/*"))]:
        if path.is_file():
            assert STRONG_KEY not in path.read_text(errors="ignore"), path
    assert "api_key_source: env:BRASSKEY_API_KEY" in (tmp_path / "brasskey.lock").read_text()


def test_a_network_address_without_a_key_stops_before_downloading(cli, tiny):
    result = cli("run", "tiny", "--host", ANY_HOST, "--yes")
    assert result.exit_code == 1
    assert "needs an API key" in result.stderr
    assert "--generate-api-key" in result.stderr
    assert tiny["hub"].downloads == []


def test_generate_api_key_shows_it_once_and_keeps_it_in_the_keychain(cli, world, monkeypatch):
    saved: list[str] = []
    monkeypatch.setattr("brasskey.cli.run.keyring_writer", lambda s, u, v: saved.append(v))
    result = cli("run", "qwen3-4b", "--dry-run", "--host", ANY_HOST, "--generate-api-key")
    assert result.exit_code == 0, result.output
    assert len(saved) == 1
    assert saved[0] in result.stderr  # shown once
    assert "saved in your keychain" in result.stderr
    assert "API key required (from your keychain)" in result.stdout
    assert "not encrypted" in result.stdout  # no TLS on the network: warned
    args = ("run", "qwen3-4b", "--dry-run", "--host", ANY_HOST, "--generate-api-key", "--json")
    document = cli(*args).json()
    assert document["data"]["security"]["generated_api_key"].startswith("bk-")


def test_next_steps_keep_the_serving_options(cli, world, monkeypatch):
    refused = cli("run", "qwen3-4b", "--dry-run", "--host", ANY_HOST, "--port", "9000")
    assert refused.exit_code == 1
    assert refused.last_line() == (
        "Next: brasskey run qwen3-4b --host 0.0.0.0 --port 9000 --dry-run --generate-api-key "
        "(create a key)"
    )
    monkeypatch.setenv("TEAM_KEY", STRONG_KEY)
    planned = cli("run", "qwen3-4b", "--dry-run", "--host", ANY_HOST, "--api-key-env", "TEAM_KEY")
    assert planned.exit_code == 0, planned.output
    assert planned.last_line() == (
        "Next: brasskey run qwen3-4b --host 0.0.0.0 --api-key-env TEAM_KEY (start it)"
    )


def test_generated_key_without_a_keychain_says_which_variable_to_export(cli, world):
    # conftest makes every keychain write fail, as on a Linux server without a secret service.
    result = cli("run", "qwen3-4b", "--dry-run", "--host", ANY_HOST, "--generate-api-key")
    assert result.exit_code == 0, result.output
    assert "export BRASSKEY_API_KEY=<this key>" in result.stderr
    assert "API key required (generated for this run)" in result.stdout


def test_generated_key_for_a_named_variable_is_not_hidden_in_the_keychain(cli, world, monkeypatch):
    saved: list[str] = []
    monkeypatch.setattr("brasskey.cli.run.keyring_writer", lambda s, u, v: saved.append(v))
    args = ("--dry-run", "--host", ANY_HOST, "--api-key-env", "TEAM_KEY", "--generate-api-key")
    result = cli("run", "qwen3-4b", *args)
    assert result.exit_code == 0, result.output
    assert saved == []
    assert "export TEAM_KEY=<this key>" in result.stderr


# --- vLLM on NVIDIA (fake engine, fake GPUs) ----------------------------------------------------


def two_l4s_with_an_old_driver_and_gpu0_busy():
    hw = machines.l4(count=2)
    gpus = tuple(
        gpu.model_copy(
            update={
                "cuda_driver_version": "12.8",
                "driver_version": "570.172.08",
                "memory_free_bytes": 2 * 2**30 if gpu.index == 0 else gpu.memory_free_bytes,
            }
        )
        for gpu in hw.accelerators
    )
    return hw.model_copy(update={"accelerators": gpus})


def test_vllm_in_docker_gets_the_freest_gpu_and_warns_about_the_driver(
    cli, tiny, monkeypatch, tmp_path
):
    import sys

    adapter = FakeVllmAdapter()
    tiny["adapter"] = adapter
    tiny["hw"] = two_l4s_with_an_old_driver_and_gpu0_busy()
    pulling = FAKE_VLLM_INSTALL.model_copy(
        update={
            "needs_setup": True,
            "setup_argv": (sys.executable, "-c", "pass"),  # stands in for `docker pull`
            "setup_note": "a 9.2 GB Docker image, one time",
        }
    )
    monkeypatch.setattr("brasskey.cli.run.detect_engines", lambda: {EngineId.VLLM: pulling})

    plan = cli("run", "tiny", "--dry-run")
    assert plan.exit_code == 0, plan.output
    assert "vLLM 0.32.0-fake in Docker (set up automatically on first run: a 9.2 GB" in plan.stdout
    assert "supports CUDA up to 12.8" in plan.stdout  # old driver: warned, not refused
    assert cli("run", "tiny", "--dry-run", "--json").json()["data"]["gpu_indices"] == [1]

    result = cli("run", "tiny", "--yes", "--save")
    assert result.exit_code == 0, result.output
    assert "Setup: vLLM (a 9.2 GB Docker image, one time)" in result.stderr
    (options,) = adapter.options
    assert options.gpu_indices == (1,)  # GPU 0 is busy
    assert options.tensor_parallel == 1
    assert tiny["chat"]["choices"]
    lock = (tmp_path / "brasskey.lock").read_text()
    assert "runtime: docker" in lock
    assert "image: vllm/vllm-openai:v0.32.0@sha256:" in lock
    assert "{model_cache}:/models/tiny:ro" in lock  # no absolute paths in the lock
    assert str(tmp_path) not in lock
