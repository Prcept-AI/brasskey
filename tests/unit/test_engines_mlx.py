# SPDX-License-Identifier: Apache-2.0
"""MLX adapter: detection (uv / PATH), launch goldens, safety defaults, failures, setup."""

import sys
from pathlib import Path

import pytest

from brasskey.engines.base import EngineInstall, LocalModel, ServeOptions
from brasskey.engines.mlx import (
    MLX_LM_VERSION,
    PINNED_FLAGS,
    RESOLVE_CUTOFF,
    MlxAdapter,
    parse_probe,
    uv_base_command,
)
from brasskey.engines.registry import refine_hardware
from brasskey.engines.types import EngineId
from brasskey.errors import BrasskeyError
from brasskey.hardware.commands import CommandOutput
from brasskey.runtime.setup import EngineSetupError, run_engine_setup
from fakes import machines
from fakes.commands import load_text
from fakes.engine import TINY_ARTIFACT

PROBE = load_text("../engines/mlx-lm-0.32.2-m3pro.probe.json")
HELP = load_text("../engines/mlx-lm-0.32.2.help")
UV = "/Users/me/.local/bin/uv"


def adapter(
    *, tools: dict[str, str], probe_ok: bool = True, help_ok: bool = True, platform="darwin"
):
    calls: list[list[str]] = []

    def run(argv, *, timeout_s=5.0):
        calls.append(list(argv))
        if "-c" in argv:
            return CommandOutput(0, PROBE) if probe_ok else CommandOutput(2, "", "offline: no env")
        return CommandOutput(0 if help_ok else 1, HELP)

    found = MlxAdapter(run=run, which=tools.get, platform_name=platform, fallback_dirs=())
    return found, calls


def test_pinned_uv_environment_that_is_ready():
    mlx_adapter, calls = adapter(tools={"uv": UV})
    install = mlx_adapter.detect()
    assert install is not None
    assert install.runtime == "uvx"
    assert install.version == "0.32.2 (mlx 0.32.4)"
    assert install.needs_setup is False
    assert install.executable == (*uv_base_command(UV), "mlx_lm.server")
    assert "--offline" in calls[0]  # detection never downloads anything
    (device,) = install.devices
    assert device.memory_total_bytes == 12_884_918_272  # Metal's working-set limit via MLX


def test_uv_environment_not_set_up_yet_needs_a_confirmed_setup():
    install = adapter(tools={"uv": UV}, probe_ok=False)[0].detect()
    assert install is not None
    assert install.needs_setup is True
    assert install.setup_note == "about 70 MB, one time"
    assert install.setup_argv[-2:] == ("mlx_lm.server", "--help")
    assert "--offline" not in install.setup_argv


def test_uv_command_is_pinned_and_reproducible():
    command = uv_base_command(UV)
    assert command[:3] == (UV, "tool", "run")
    assert "--managed-python" in command
    assert command[command.index("--exclude-newer") + 1] == RESOLVE_CUTOFF
    assert command[-1] == f"mlx-lm=={MLX_LM_VERSION}"


def test_mlx_lm_already_on_path(tmp_path):
    python = tmp_path / "python3"
    python.write_text("")
    server = tmp_path / "mlx_lm.server"
    server.write_text(f"#!{python}\nimport mlx_lm\n")
    mlx_adapter, calls = adapter(tools={"mlx_lm.server": str(server), "uv": UV})
    install = mlx_adapter.detect()
    assert install is not None
    assert install.runtime == "process"
    assert install.executable == (str(server),)
    assert calls[0][0] == str(python)  # probed with the script's own interpreter
    assert "--decode-concurrency" in install.flags


def test_broken_path_install_is_not_used(tmp_path):
    server = tmp_path / "mlx_lm.server"
    server.write_text("#!/usr/bin/env python3\n")
    assert adapter(tools={"mlx_lm.server": str(server)}, help_ok=False)[0].detect() is None


def test_nothing_to_run_mlx_with():
    assert adapter(tools={})[0].detect() is None


def test_not_on_a_mac():
    mlx_adapter, calls = adapter(tools={"uv": UV}, platform="linux")
    assert mlx_adapter.detect() is None
    assert calls == []


def test_parse_probe_edge_cases():
    assert parse_probe("Downloading x\nnot json") is None
    version, devices = parse_probe('{"mlx_lm": "1.0", "mlx": "2.0"}') or ("", ())
    assert version == "1.0 (mlx 2.0)"
    assert devices == ()


def install(flags=PINNED_FLAGS) -> EngineInstall:
    return EngineInstall(
        engine=EngineId.MLX,
        version="0.32.2 (mlx 0.32.4)",
        runtime="uvx",
        executable=(*uv_base_command(UV), "mlx_lm.server"),
        flags=flags,
    )


def local() -> LocalModel:
    snapshot = Path("/cache/snapshots/abc")
    return LocalModel(artifact=TINY_ARTIFACT, snapshot_dir=snapshot, files=(snapshot / "x",))


def options(**overrides):
    values = {
        "port": 8000,
        "served_model_name": "qwen3-8b",
        "context_length": 8192,
        "cache_budget_bytes": 1_207_959_552,
    }
    return ServeOptions(**(values | overrides))


def test_launch_golden_with_safe_defaults():
    spec = MlxAdapter().build_launch(install(), local(), options(), machines.m3_pro_18gb())
    assert spec.argv[: len(uv_base_command(UV)) + 1] == (*uv_base_command(UV), "mlx_lm.server")
    assert spec.argv[len(uv_base_command(UV)) + 1 :] == (
        "--model", "qwen3-8b",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--allowed-origins", "http://127.0.0.1:8000,http://localhost:8000",
        "--log-level", "INFO",
        "--decode-concurrency", "1",
        "--prompt-concurrency", "1",
        "--prompt-cache-bytes", "1207959552",
    )  # fmt: skip
    assert spec.health_url == "http://127.0.0.1:8000/health"
    assert spec.env["HF_HUB_OFFLINE"] == "1"
    assert spec.cwd_links == {"qwen3-8b": Path("/cache/snapshots/abc")}


def test_launch_without_optional_flags_on_older_versions():
    spec = MlxAdapter().build_launch(
        install(frozenset({"--model", "--host", "--port"})),
        local(),
        options(),
        machines.m3_pro_18gb(),
    )
    assert "--allowed-origins" not in spec.argv
    assert "--prompt-cache-bytes" not in spec.argv


def test_api_keys_are_refused():
    with pytest.raises(BrasskeyError) as caught:
        MlxAdapter().build_launch(
            install(), local(), options(api_key_env="KEY"), machines.m3_pro_18gb()
        )
    assert "can't require an API key" in caught.value.message
    assert MlxAdapter.supports_api_key is False


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("[METAL] Command buffer execution failed: Insufficient Memory", "ran out of memory"),
        ("[metal::malloc] Attempting to allocate 9999999 bytes", "ran out of memory"),
        ("OSError: [Errno 48] Address already in use", "already in use"),
        ("ValueError: Model type qwen9 not supported.", "doesn't support this model"),
        ("ModuleNotFoundError: No module named 'mlx'", "environment is incomplete"),
    ],
)
def test_explain_failure(line, message):
    error = MlxAdapter().explain_failure(1, [line])
    assert error is not None
    assert message in error.message


def test_model_type_needs_both_fragments():
    assert MlxAdapter().explain_failure(1, ["model type: dense"]) is None


def test_install_hints():
    assert MlxAdapter().install_hint(machines.m3_pro_18gb()).commands[0] == "brew install uv"
    assert MlxAdapter().install_hint(machines.h100()).commands == ()


def test_mlx_measurement_replaces_the_estimate():
    found = adapter(tools={"uv": UV})[0].detect()
    refined = refine_hardware(machines.m3_pro_18gb(), {EngineId.MLX: found})
    assert refined.accelerators[0].memory_total_bytes == 12_884_918_272
    assert refined.accelerators[0].memory_is_estimate is False


def test_fallback_dirs_are_searched(tmp_path):
    (tmp_path / "uv").write_text("")
    found = MlxAdapter(
        run=lambda argv, *, timeout_s=5.0: CommandOutput(0, PROBE),
        which=lambda _name: None,
        platform_name="darwin",
        fallback_dirs=(tmp_path,),
    ).detect()
    assert found is not None
    assert found.executable[0] == str(tmp_path / "uv")


# --- one-time setup -------------------------------------------------------------------------------


def setup_install(argv: tuple[str, ...]) -> EngineInstall:
    return install().model_copy(update={"needs_setup": True, "setup_argv": argv})


def test_setup_runs_the_setup_command():
    run_engine_setup(setup_install((sys.executable, "-c", "print('ready')")))


def test_setup_failure_shows_the_last_line():
    argv = (
        sys.executable,
        "-c",
        "import sys; print('resolving', file=sys.stderr); "
        "print('error: no network', file=sys.stderr); sys.exit(2)",
    )
    with pytest.raises(EngineSetupError) as caught:
        run_engine_setup(setup_install(argv))
    assert "error: no network" in caught.value.message


def test_setup_timeout_and_missing_tool():
    with pytest.raises(EngineSetupError, match="longer than"):
        run_engine_setup(
            setup_install((sys.executable, "-c", "import time; time.sleep(5)")), timeout_s=0.3
        )
    with pytest.raises(EngineSetupError, match="wasn't found"):
        run_engine_setup(setup_install(("/nonexistent/uv", "tool", "run")))


def test_nothing_to_set_up_is_a_no_op():
    run_engine_setup(install())
