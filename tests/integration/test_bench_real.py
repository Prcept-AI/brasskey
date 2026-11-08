# SPDX-License-Identifier: Apache-2.0
"""The real thing: ``brasskey bench`` with real GuideLLM against a real llama.cpp server.

Opt-in (``uv run pytest -m engine``): needs ``llama-server`` and uv. The first run sets up the
pinned GuideLLM (about 300 MB) and uses Qwen3 0.6B (639 MB, shared with the llama.cpp test).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from fakes.helpers import free_port, read_line

pytestmark = pytest.mark.engine

KEY = "bk-bench-integration-Lq9XzP4mVw8RtY6nJ3aK"


@pytest.mark.skipif(shutil.which("llama-server") is None, reason="llama.cpp is not installed")
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv for the pinned GuideLLM")
def test_bench_a_real_server_with_real_guidellm(tmp_path: Path) -> None:
    home = tmp_path / "brasskey-home"
    env = {**os.environ, "BRASSKEY_HOME": str(home), "NO_COLOR": "1", "BRASSKEY_API_KEY": KEY}
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "brasskey", "run", "qwen3-0.6b", "--engine", "llamacpp",
         "--yes", "--json", "--port", str(port)],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert server.stdout is not None
        ready = json.loads(read_line(server.stdout, timeout_s=1800))["data"]
        assert ready["status"] == "ready"
        bench = subprocess.run(
            [sys.executable, "-m", "brasskey", "bench", ready["base_url"], "--repeats", "2",
             "--duration", "5", "--warmup", "0", "--concurrency", "1", "--prompt-tokens", "64",
             "--output-tokens", "32", "--yes", "--json", "--save", "results.json"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1200, check=False,
        )  # fmt: skip
        assert bench.returncode == 0, bench.stderr[-3000:]
        data = json.loads(bench.stdout)["data"]
        throughput = next(
            m for m in data["candidate"]["metrics"] if m["key"] == "output_tokens_per_s"
        )
        assert throughput["mean"] > 1
        assert len(throughput["values"]) == 2
        assert data["candidate"]["engine"] == "llama.cpp"
        saved = (tmp_path / "results.json").read_text()
        assert KEY not in saved
        assert KEY not in bench.stdout
        assert "request_args" not in saved  # numbers only
        assert not list((home / "state" / "bench").glob("*/*")), "GuideLLM files left behind"
        for log in (home / "logs").glob("bench-*.log"):
            assert KEY not in log.read_text(errors="ignore")
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
