# SPDX-License-Identifier: Apache-2.0
"""The real thing: ``brasskey run`` with real llama.cpp and a real small model.

Opt-in (``uv run pytest -m engine``): needs ``llama-server`` installed and downloads Qwen3 0.6B
(Q8_0, 639 MB) into the Hugging Face cache on first run.
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

from fakes.helpers import free_port, read_line

pytestmark = pytest.mark.engine

MARKER = "brasskey-privacy-marker-7f3a"
CLIENT_KEY = "bk-integration-Q7wE9rT2yU4iO6pA8sD0fG1h"


@pytest.mark.skipif(shutil.which("llama-server") is None, reason="llama.cpp is not installed")
def test_run_qwen3_0_6b_with_llama_cpp(tmp_path):
    home = tmp_path / "brasskey-home"
    # A client key on loopback: the Brasskey proxy must require it (and keep it out of argv,
    # logs, state files and the lock).
    env = {**os.environ, "BRASSKEY_HOME": str(home), "NO_COLOR": "1",
           "BRASSKEY_API_KEY": CLIENT_KEY}  # fmt: skip
    port = free_port()
    command = [sys.executable, "-m", "brasskey", "run", "qwen3-0.6b", "--engine", "llamacpp",
               "--yes", "--json", "--port", str(port)]  # fmt: skip
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        ready = json.loads(read_line(process.stdout, timeout_s=1800))  # first run downloads
        data = ready["data"]
        assert data["status"] == "ready"
        assert data["base_url"] == f"http://127.0.0.1:{port}/v1"
        assert data["auth"] == "api_key"
        engine_pid = data["pid"]
        engine_port = data["engine_port"]
        assert engine_port != port
        assert CLIENT_KEY not in " ".join(psutil.Process(engine_pid).cmdline())

        auth = {"Authorization": f"Bearer {CLIENT_KEY}"}
        with httpx2.Client(timeout=120, trust_env=False, headers=auth) as http:
            assert httpx2.get(f"{data['base_url']}/models", trust_env=False).status_code == 401
            direct = httpx2.get(f"http://127.0.0.1:{engine_port}/v1/models", trust_env=False)
            assert direct.status_code == 401  # the engine itself demands the internal key
            models = http.get(f"{data['base_url']}/models").json()
            assert any(entry["id"] == "qwen3-0.6b" for entry in models["data"])
            reply = http.post(
                f"{data['base_url']}/chat/completions",
                json={
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": f"Say hi. ({MARKER}) /no_think"}],
                    "max_tokens": 32,
                },
            ).json()
            assert reply["choices"][0]["message"]["content"] is not None
            assert http.get(f"http://127.0.0.1:{port}/metrics").status_code == 200

        doctor = subprocess.run(
            [sys.executable, "-m", "brasskey", "doctor", "--json"],
            env=env, capture_output=True, text=True, check=True, timeout=60,
        )  # fmt: skip
        servers = json.loads(doctor.stdout)["data"]["servers"]
        assert [server["port"] for server in servers] == [port]

        process.send_signal(signal.SIGINT)  # what Ctrl-C does
        assert process.wait(timeout=60) == 0
        deadline = time.monotonic() + 15
        while psutil.pid_exists(engine_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not psutil.pid_exists(engine_pid), "llama-server outlived brasskey"
        assert not list((home / "state" / "servers").glob("*.json"))
        log = Path(data["log_path"]).read_text(errors="replace")
        assert MARKER not in log, "the prompt reached the engine log"
        for path in [*home.rglob("*"), *tmp_path.glob("brasskey.*")]:
            if path.is_file():
                assert CLIENT_KEY not in path.read_text(errors="ignore"), path
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
