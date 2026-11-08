# SPDX-License-Identifier: Apache-2.0
"""``brasskey bench`` end to end with fake endpoints and a fake GuideLLM (no network, no GPU)."""

import json
from dataclasses import dataclass, field

import httpx2
import pytest

from brasskey.bench.guidellm import GuidellmInstall, RunTarget
from brasskey.bench.models import METRIC_OPTIONS, RunMetrics, Workload
from brasskey.cli.bench import MetricOption
from brasskey.config.io import read_lock, write_lock
from brasskey.engines.types import EngineId
from brasskey.models.catalog import load_catalog
from brasskey.planner import RunRequest, plan_run
from brasskey.planner.replay import lock_from_plan
from fakes import machines
from fakes.helpers import fake_api_key
from fakes.hub import FakeHub

CANDIDATE = "http://127.0.0.1:8812/v1"
BASELINE = "http://127.0.0.1:8811/v1"
KEY = fake_api_key("bench")
READY = GuidellmInstall(version="0.8.1", runtime="uvx", executable=("uv", "tool", "run"))


@dataclass
class World:
    # Throughput per server root, run by run (index = repeat); latency follows inversely.
    throughput: dict[str, list[float]] = field(default_factory=dict)
    models: dict[str, list[str]] = field(default_factory=dict)
    needs_key: set[str] = field(default_factory=set)
    calls: list[tuple[str, int, int, RunTarget]] = field(default_factory=list)
    setups: list[tuple[str, ...]] = field(default_factory=list)
    install: GuidellmInstall | None = READY
    hub: FakeHub | None = None

    def run(self, install, target, workload, *, seed, duration_s, work_dir, log_path):  # type: ignore[no-untyped-def]
        assert work_dir.is_dir()
        assert oct(work_dir.stat().st_mode & 0o777) == "0o700"  # private while GuideLLM runs
        self.calls.append((target.target, seed, duration_s, target))
        values = self.throughput[target.target]
        measured = [call for call in self.calls if call[0] == target.target and call[2] > 10]
        value = values[(len(measured) - 1) % len(values)] if duration_s > 10 else values[0]
        return RunMetrics(
            seed=seed, duration_s=duration_s, successful=100, errored=0, incomplete=1,
            output_tokens_per_s=value, requests_per_s=value / 32, ttft_ms_p50=30000 / value,
            ttft_ms_p95=40000 / value, itl_ms_p50=900 / value, latency_s_p50=60 / value,
            latency_s_p95=80 / value,
        )  # fmt: skip


@pytest.fixture
def world(monkeypatch, tmp_path):
    state = World()
    state.models = {CANDIDATE: ["qwen3-0.6b"], BASELINE: ["qwen3-0.6b"]}
    hub = FakeHub(cache_dir=tmp_path / "hf")
    model = next(m for m in load_catalog().models if m.id == "qwen3-0.6b")
    tokenizer = next(a for a in model.artifacts if a.format.value == "mlx")
    for file in tokenizer.files:
        if not file.path.endswith(".safetensors"):
            hub.add_cached_blob(tokenizer.repo_id, file.path, b"x", file.sha256 or "0" * 64)
    state.hub = hub

    def handler(request: httpx2.Request) -> httpx2.Response:
        base = str(request.url).removesuffix("/models")
        if base not in state.models:
            raise httpx2.ConnectError("refused", request=request)
        if base in state.needs_key and request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx2.Response(401, json={"error": {"code": "invalid_api_key"}})
        return httpx2.Response(200, json={"data": [{"id": m} for m in state.models[base]]})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "brasskey.cli.bench.make_http_client",
        lambda: httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setattr("brasskey.cli.bench.detect_guidellm_install", lambda: state.install)
    monkeypatch.setattr("brasskey.cli.bench.make_hub", lambda: state.hub)
    monkeypatch.setattr("brasskey.cli.bench.run_guidellm_step", state.run)
    monkeypatch.setattr(
        "brasskey.cli.bench.run_setup",
        lambda argv, *, name, timeout_s: state.setups.append(tuple(argv)),
    )
    return state


SHORT = ("--duration", "20", "--warmup", "5", "--yes")


def last_stdout_line(result) -> str:  # type: ignore[no-untyped-def]
    return result.stdout.strip().splitlines()[-1]


def root(url: str) -> str:
    return url.removesuffix("/v1")


def test_single_endpoint(cli, world):
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    result = cli("bench", CANDIDATE, *SHORT)
    assert result.exit_code == 0, result.output
    assert "Benchmark of qwen3-0.6b at http://127.0.0.1:8812/v1" in result.stdout
    assert "95% interval" in result.stdout
    assert "Output tokens/s" in result.stdout
    # The plan line goes to stderr before the run; the report and Next end stdout.
    assert last_stdout_line(result).startswith(
        "Next: brasskey bench http://127.0.0.1:8812/v1 --save"
    )
    assert [call[1] for call in world.calls] == [1042, 42, 43, 44]  # warm-up, then 3 seeds
    document = cli("bench", CANDIDATE, *SHORT, "--json").json()
    metrics = {m["key"]: m for m in document["data"]["candidate"]["metrics"]}
    assert metrics["output_tokens_per_s"]["mean"] == pytest.approx(130.333, abs=1e-3)
    assert (
        metrics["output_tokens_per_s"]["ci_low"]
        < 130.333
        < metrics["output_tokens_per_s"]["ci_high"]
    )
    assert document["data"]["comparison"] is None
    assert document["data"]["tool"] == "guidellm 0.8.1"


@pytest.mark.parametrize(
    ("candidate", "exit_code", "verdict"),
    [
        ([130.0, 128.0, 133.0], 0, "PASS"),
        ([80.0, 82.0, 79.0], 3, "FAIL"),
        ([104.0, 96.0, 118.0], 4, "INCONCLUSIVE"),
    ],
)
def test_compare_verdicts_and_exit_codes(cli, world, candidate, exit_code, verdict):
    world.throughput = {root(CANDIDATE): candidate, root(BASELINE): [100.0, 115.0, 90.0]
                        if verdict == "INCONCLUSIVE" else [100.0, 102.0, 98.0]}  # fmt: skip
    result = cli("bench", CANDIDATE, "--compare", BASELINE, *SHORT)
    assert result.exit_code == exit_code, result.output
    assert f"Verdict: {verdict}" in result.stdout
    assert "Baseline   qwen3-0.6b at http://127.0.0.1:8811/v1" in result.stdout
    if verdict == "INCONCLUSIVE":
        assert "--repeats 5" in last_stdout_line(result)


def test_compare_alternates_endpoints_with_the_same_seeds(cli, world):
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0], root(BASELINE): [100, 102, 98]}
    cli("bench", CANDIDATE, "--compare", BASELINE, *SHORT)
    order = [(call[0].rsplit(":", 1)[1], call[1]) for call in world.calls]
    assert order == [
        ("8812", 1042), ("8811", 1042),
        ("8812", 42), ("8811", 42), ("8811", 43), ("8812", 43), ("8812", 44), ("8811", 44),
    ]  # fmt: skip
    tokenizers = {call[3].tokenizer for call in world.calls}
    assert len(tokenizers) == 1  # identical prompts for both endpoints


def test_save_then_compare_against_the_saved_results(cli, world, tmp_path):
    world.throughput = {root(BASELINE): [100.0, 102.0, 98.0]}
    saved = cli("bench", BASELINE, *SHORT, "--save", "base.json")
    assert saved.exit_code == 0, saved.output
    text = (tmp_path / "base.json").read_text()
    assert json.loads(text)["format"] == "brasskey.bench/v1"
    world.calls.clear()
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    result = cli("bench", CANDIDATE, "--baseline", "base.json", "--yes")
    assert result.exit_code == 0, result.output
    assert "(saved results)" in result.stdout
    assert {call[0] for call in world.calls} == {root(CANDIDATE)}  # the baseline isn't rerun
    assert {call[2] for call in world.calls} == {5, 20}  # the saved workload was reused


def test_a_different_workload_than_the_saved_baseline_is_refused(cli, world):
    world.throughput = {root(BASELINE): [100.0, 102.0, 98.0], root(CANDIDATE): [130.0]}
    cli("bench", BASELINE, *SHORT, "--save", "base.json")
    result = cli("bench", CANDIDATE, "--baseline", "base.json", "--duration", "30", "--yes")
    assert result.exit_code == 1
    assert "different workload (duration_s)" in result.stderr
    assert result.last_line() == (
        "Next: brasskey bench http://127.0.0.1:8812/v1 --baseline base.json (reuse the saved "
        "workload)"
    )


def test_compare_and_baseline_together_is_a_usage_error(cli, world):
    result = cli("bench", CANDIDATE, "--compare", BASELINE, "--baseline", "x.json")
    assert result.exit_code == 2


def test_keys_come_from_the_environment_and_are_never_written(cli, world, monkeypatch, tmp_path):
    world.needs_key = {CANDIDATE}
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    refused = cli("bench", CANDIDATE, *SHORT)
    assert refused.exit_code == 1
    assert "needs an API key" in refused.stderr
    monkeypatch.setenv("BRASSKEY_API_KEY", KEY)
    result = cli("bench", CANDIDATE, *SHORT, "--save", "out.json", "--json")
    assert result.exit_code == 0, result.output
    assert world.calls[-1][3].api_key is not None
    assert KEY not in result.stdout
    assert KEY not in (tmp_path / "out.json").read_text()


def test_nothing_runs_without_confirmation(cli, world):
    world.throughput = {root(CANDIDATE): [130.0]}
    result = cli("bench", CANDIDATE)  # tests have no terminal and no --yes
    assert result.exit_code == 1
    assert "Benchmark http://127.0.0.1:8812/v1 for about" in result.stderr
    assert world.calls == []


def test_guidellm_setup_is_part_of_the_confirmed_plan(cli, world):
    world.install = GuidellmInstall(
        version="0.8.1", runtime="uvx", executable=("uv",), needs_setup=True,
        setup_argv=("uv", "tool", "run", "guidellm", "--version"), setup_note="about 300 MB",
    )  # fmt: skip
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    question = cli("bench", CANDIDATE)
    assert "Set up GuideLLM 0.8.1 (about 300 MB)" in question.stderr
    assert world.setups == []
    result = cli("bench", CANDIDATE, *SHORT)
    assert result.exit_code == 0, result.output
    assert world.setups == [("uv", "tool", "run", "guidellm", "--version")]


def test_missing_guidellm_points_at_uv(cli, world):
    world.install = None
    result = cli("bench", CANDIDATE, *SHORT)
    assert result.exit_code == 1
    assert "GuideLLM isn't available" in result.stderr
    assert result.last_line() == "Next: brew install uv (install uv)"


def test_tokenizer_files_are_downloaded_when_not_cached(cli, world, tmp_path):
    world.hub = FakeHub(cache_dir=tmp_path / "empty-cache")
    question = cli("bench", CANDIDATE)
    assert "Download the tokenizer (" in question.stderr


def test_lock_collects_benchmarks(cli, world, tmp_path):
    hw = machines.m3_pro_18gb()
    plan = plan_run(RunRequest(model="qwen3-0.6b"), hw, load_catalog(),
                    runnable=[EngineId.LLAMACPP])  # fmt: skip
    write_lock(tmp_path, lock_from_plan(plan, hw, intent=None, host="127.0.0.1", port=8812))
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    assert cli("bench", CANDIDATE, *SHORT, "--lock").exit_code == 0
    assert cli("bench", CANDIDATE, *SHORT, "--lock").exit_code == 0
    lock = read_lock(tmp_path)
    assert lock is not None
    assert len(lock.benchmarks) == 2
    assert lock.benchmarks[0].workload == Workload(duration_s=20, warmup_s=5)
    world.models[CANDIDATE] = ["qwen3-4b"]
    world.calls.clear()
    other = cli("bench", CANDIDATE, *SHORT, "--lock")
    assert other.exit_code == 1
    assert "is for qwen3-0.6b" in other.stderr
    assert world.calls == []  # refused before spending any time on runs


def test_lock_needs_a_lock_file(cli, world):
    world.throughput = {root(CANDIDATE): [130.0, 128.0, 133.0]}
    result = cli("bench", CANDIDATE, *SHORT, "--lock")
    assert result.exit_code == 1
    assert "There is no brasskey.lock" in result.stderr
    assert world.calls == []
    bad_save = cli("bench", CANDIDATE, *SHORT, "--save", "missing/dir/out.json")
    assert bad_save.exit_code == 2
    assert world.calls == []


def test_metric_options_match_the_metric_table():
    assert tuple(option.value for option in MetricOption) == METRIC_OPTIONS
