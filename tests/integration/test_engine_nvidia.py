# SPDX-License-Identifier: Apache-2.0
"""The real thing on NVIDIA: ``brasskey run --engine vllm|sglang`` with Qwen3 0.6B.

Opt-in (``uv run pytest -m gpu``), meant for a Linux machine with an NVIDIA GPU such as an AWS
g6.xlarge (docs/gpu-testing.md). Each engine runs through whatever Brasskey detects: ``vllm`` /
``sglang`` on PATH, or the pinned Docker image (pulled on first use: 9.2 GB for vLLM, 14.4 GB
for SGLang). Engines that aren't available are skipped with the reason.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import psutil
import pytest

from brasskey.engines.registry import adapter_for
from brasskey.engines.types import EngineId
from fakes.helpers import free_port, read_line

pytestmark = pytest.mark.gpu

MARKER = "brasskey-privacy-marker-gpu-5d2e"
CLIENT_KEY = "bk-gpu-integration-Lq9XzP4mVw8RtY6nJ3aK"
FIRST_RUN_TIMEOUT_S = 3600  # image pull + model download + start


def running_containers(name_filter: str) -> list[str]:
    docker = shutil.which("docker")
    if docker is None:
        return []
    out = subprocess.run(
        [docker, "ps", "-a", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False, timeout=30,
    )  # fmt: skip
    return out.stdout.split()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="NVIDIA engines need Linux")
@pytest.mark.parametrize("engine", [EngineId.VLLM, EngineId.SGLANG])
def test_run_qwen3_0_6b_on_nvidia(engine: EngineId, tmp_path: Path) -> None:
    adapter = adapter_for(engine)
    assert adapter is not None
    install = adapter.detect()
    if install is None:
        pytest.skip(f"{engine.display_name}: not on PATH and no Docker with the NVIDIA toolkit")

    home = tmp_path / "brasskey-home"
    env = {**os.environ, "BRASSKEY_HOME": str(home), "NO_COLOR": "1",
           "BRASSKEY_API_KEY": CLIENT_KEY}  # fmt: skip
    port = free_port()
    command = [sys.executable, "-m", "brasskey", "run", "qwen3-0.6b", "--engine", engine.value,
               "--yes", "--json", "--port", str(port), "--save"]  # fmt: skip
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        data = json.loads(read_line(process.stdout, timeout_s=FIRST_RUN_TIMEOUT_S))["data"]
        assert data["status"] == "ready", data
        assert data["engine"] == engine.value
        assert data["engine_runtime"] == install.runtime
        assert data["gpu_indices"]
        base = data["base_url"]
        engine_port = data["engine_port"]

        auth = {"Authorization": f"Bearer {CLIENT_KEY}"}
        with httpx2.Client(timeout=300, trust_env=False, headers=auth) as http:
            assert httpx2.get(f"{base}/models", trust_env=False).status_code == 401
            models = http.get(f"{base}/models").json()
            assert [entry["id"] for entry in models["data"]] == ["qwen3-0.6b"]
            chat = {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": f"Say hi. ({MARKER}) /no_think"}],
                "max_tokens": 32,
            }
            reply = http.post(f"{base}/chat/completions", json=chat).json()
            assert reply["choices"][0]["message"]["content"] is not None
            events = 0
            with http.stream("POST", f"{base}/chat/completions",
                             json={**chat, "stream": True}) as streamed:  # fmt: skip
                for line in streamed.iter_lines():
                    events += line.startswith("data: ") and line != "data: [DONE]"
            assert events > 1
            other = http.post(f"{base}/chat/completions", json={**chat, "model": "other"})
            assert other.status_code == 404
            metrics = http.get(f"{base.removesuffix('/v1')}/metrics").text
            assert f"{engine.value}:" in metrics  # vllm:... / sglang:... Prometheus series
            direct = httpx2.get(f"http://127.0.0.1:{engine_port}/v1/models", trust_env=False)
            if adapter.supports_api_key:
                assert direct.status_code == 401  # the engine demands Brasskey's internal key

        if install.runtime == "docker":
            name = f"brasskey-{engine.value}-{engine_port}"
            assert running_containers(name) == [name]

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=120) == 0
        deadline = time.monotonic() + 30
        while psutil.pid_exists(data["pid"]) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert not psutil.pid_exists(data["pid"]), "the engine outlived brasskey"
        if install.runtime == "docker":
            assert running_containers(f"brasskey-{engine.value}-{engine_port}") == []
        assert not list((home / "state" / "servers").glob("*.json"))
        assert MARKER not in Path(data["log_path"]).read_text(errors="replace")
        lock = (tmp_path / "brasskey.lock").read_text()
        assert f"runtime: {install.runtime}" in lock
        assert str(Path.home()) not in lock
        for path in [*home.rglob("*"), *tmp_path.glob("brasskey.*")]:
            if path.is_file():
                assert CLIENT_KEY not in path.read_text(errors="ignore"), path
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
