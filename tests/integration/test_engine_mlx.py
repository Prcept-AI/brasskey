# SPDX-License-Identifier: Apache-2.0
"""The real thing on Apple Silicon: ``brasskey run --engine mlx`` with a real small model.

Opt-in (``uv run pytest -m engine``): needs an Apple Silicon Mac with uv (Brasskey sets up the
pinned mlx-lm environment on first use) and downloads Qwen3 0.6B in MLX 4-bit (about 330 MB).
"""

import json
import os
import platform
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

MARKER = "brasskey-privacy-marker-mlx-41c9"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


@pytest.mark.skipif(not APPLE_SILICON, reason="MLX needs an Apple Silicon Mac")
@pytest.mark.skipif(
    shutil.which("uv") is None and shutil.which("mlx_lm.server") is None,
    reason="needs uv (or mlx-lm) to run MLX",
)
def test_run_qwen3_0_6b_with_mlx(tmp_path):
    home = tmp_path / "brasskey-home"
    env = {**os.environ, "BRASSKEY_HOME": str(home), "NO_COLOR": "1"}
    port = free_port()
    command = [sys.executable, "-m", "brasskey", "run", "qwen3-0.6b", "--engine", "mlx",
               "--yes", "--json", "--port", str(port)]  # fmt: skip
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        data = json.loads(read_line(process.stdout, timeout_s=1800))["data"]
        assert data["status"] == "ready"
        assert data["engine"] == "mlx"
        base = data["base_url"]

        with httpx2.Client(timeout=120, trust_env=False) as http:
            reply = http.post(
                f"{base}/chat/completions",
                json={
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": f"Say hi. ({MARKER}) /no_think"}],
                    "max_tokens": 32,
                },
            ).json()
            assert reply["choices"][0]["message"]["content"] is not None

            # Only the served model is visible, even though mlx_lm.server would load any name.
            models = http.get(f"{base}/models").json()
            assert [entry["id"] for entry in models["data"]] == ["qwen3-0.6b"]
            other = http.post(f"{base}/chat/completions", json={"model": "x/y", "messages": []})
            assert other.status_code == 404

            # A random web page in a browser is refused by the proxy...
            evil = http.get(f"{base}/models", headers={"Origin": "https://evil.example"})
            assert evil.status_code == 403
            assert "access-control-allow-origin" not in evil.headers
            # ...while the server's own origin is allowed.
            own = http.get(f"{base}/models", headers={"Origin": f"http://127.0.0.1:{port}"})
            assert own.status_code == 200

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=60) == 0
        deadline = time.monotonic() + 15
        while psutil.pid_exists(data["pid"]) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not psutil.pid_exists(data["pid"]), "mlx_lm.server outlived brasskey"
        assert not list((home / "state" / "servers").glob("*.json"))
        assert MARKER not in Path(data["log_path"]).read_text(errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
