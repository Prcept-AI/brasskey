# SPDX-License-Identifier: Apache-2.0
"""Benchmark building blocks: statistics, GuideLLM (detection, argv, report), endpoints,
tokenizers and run order. The CLI is tested in test_cli_bench.py."""

import json
import math
import sys
from pathlib import Path

import httpx2
import pytest

from brasskey.bench.analysis import compare, summarize
from brasskey.bench.endpoint import (
    Endpoint,
    brasskey_server,
    choose_model,
    parse_endpoint,
    served_models,
)
from brasskey.bench.guidellm import (
    KEY_ENV,
    BenchError,
    GuidellmInstall,
    ReportFormatError,
    RunTarget,
    build_argv,
    detect_guidellm,
    parse_report,
    run_environment,
    run_once,
    uv_base_command,
)
from brasskey.bench.models import EndpointResult, RunMetrics, Workload
from brasskey.bench.runner import estimate_seconds, execute, schedule
from brasskey.bench.stats import (
    Verdict,
    compare_runs,
    mean_interval,
    student_t_cdf,
    t_quantile,
)
from brasskey.bench.tokenizer import choose_tokenizer, fetch_tokenizer
from brasskey.hardware.commands import CommandOutput
from brasskey.models.catalog import load_catalog
from brasskey.security.apikeys import ApiKey
from fakes.helpers import fake_api_key
from fakes.hub import FakeHub

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "bench"
REPORT = (FIXTURES / "guidellm-0.8.1-report.json").read_text()
FAKE_GUIDELLM = Path(__file__).resolve().parent.parent / "fakes" / "fake_guidellm.py"
KEY = ApiKey(fake_api_key("bench"), "env:BRASSKEY_API_KEY")


# --- statistics ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 1, 12.7062047362), (0.975, 2, 4.3026527297), (0.975, 4, 2.7764451052),
        (0.975, 9, 2.2621571628), (0.975, 30, 2.0422724563), (0.995, 5, 4.0321429836),
        (0.95, 10, 1.8124611228), (0.025, 4, -2.7764451052), (0.975, math.inf, 1.9599639845),
    ],
)  # fmt: skip
def test_t_quantiles_match_published_tables(p, df, expected):
    assert t_quantile(p, df) == pytest.approx(expected, abs=1e-6)


def test_t_cdf_and_quantile_are_inverse_for_fractional_df():
    for df in (1.5, 3.7, 12.25):
        assert student_t_cdf(t_quantile(0.9, df), df) == pytest.approx(0.9, abs=1e-9)


def test_mean_interval():
    interval = mean_interval([130.4, 128.0, 133.1])
    assert interval.mean == pytest.approx(130.5)
    assert interval.low == pytest.approx(124.1618, abs=1e-3)
    assert interval.high == pytest.approx(136.8382, abs=1e-3)
    assert interval.cv == pytest.approx(0.01955, abs=1e-4)
    single = mean_interval([5.0])
    assert single.low is None
    assert single.cv is None


def test_clear_improvement_passes_and_clear_regression_fails():
    better = compare_runs([100, 102, 98], [120, 118, 123], better="higher")
    assert not isinstance(better, str)
    assert better.verdict == Verdict.PASS
    assert better.change == pytest.approx(0.2033, abs=1e-3)
    worse = compare_runs([100, 102, 98], [80, 82, 79], better="higher")
    assert not isinstance(worse, str)
    assert worse.verdict == Verdict.FAIL


def test_overlapping_runs_are_inconclusive():
    result = compare_runs([100, 115, 90], [104, 96, 118], better="higher")
    assert not isinstance(result, str)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert "can't tell" in result.reason


def test_lower_is_better_for_latency():
    faster = compare_runs([300, 305, 298], [250, 252, 249], better="lower")
    assert not isinstance(faster, str)
    assert faster.verdict == Verdict.PASS
    assert faster.change < 0  # the latency went down: that's the improvement


def test_margin_and_not_worse():
    # 3% faster: not "better by 5%", but certainly "no more than 5% worse".
    base, cand = [100.0, 100.5, 99.5], [103.0, 103.4, 102.6]
    by_five = compare_runs(base, cand, better="higher", goal="better", margin=0.05)
    within = compare_runs(base, cand, better="higher", goal="not-worse", margin=0.05)
    assert not isinstance(by_five, str)
    assert not isinstance(within, str)
    assert by_five.verdict == Verdict.FAIL
    assert within.verdict == Verdict.PASS
    assert "no more than 5% worse" in within.reason


def test_cases_without_an_interval_say_why():
    assert "two runs" in str(compare_runs([1.0], [2.0, 2.1], better="higher"))
    assert "zero" in str(compare_runs([1.0, 0.0], [2.0, 2.1], better="higher"))
    coarse = compare_runs([1.0, 1.0, 1.0], [1.25, 1.25, 1.25], better="higher")
    assert isinstance(coarse, str)
    assert "too coarse" in coarse
    with pytest.raises(ValueError, match="margin"):
        compare_runs([1.0, 1.1], [1.0, 1.1], better="higher", margin=1.5)


# --- GuideLLM --------------------------------------------------------------------------------


def test_parse_a_real_guidellm_0_8_1_report():
    run = parse_report(REPORT, seed=7)
    assert run.output_tokens_per_s == pytest.approx(130.3966, abs=1e-3)
    assert run.requests_per_s == pytest.approx(4.0)
    assert run.ttft_ms_p50 == pytest.approx(285.841, abs=1e-3)
    assert run.ttft_ms_p95 == pytest.approx(300.843, abs=1e-3)
    assert run.itl_ms_p50 == pytest.approx(6.3294, abs=1e-3)
    assert run.latency_s_p95 == pytest.approx(0.50136, abs=1e-4)
    assert (run.successful, run.errored, run.incomplete) == (33, 0, 1)
    assert run.seed == 7


def test_report_has_no_request_text():
    """sample_size=0: GuideLLM kept no prompts or answers in the fixture's run."""
    requests = json.loads(REPORT)["benchmarks"][0]["requests"]["successful"]
    assert all(entry["request_args"] is None and entry["output"] is None for entry in requests)


def test_unknown_report_formats_fail_loudly():
    data = json.loads(REPORT)
    data["metadata"]["version"] = 3
    with pytest.raises(ReportFormatError, match="format 3"):
        parse_report(json.dumps(data), seed=1)
    del data["benchmarks"][0]["metrics"]["inter_token_latency_ms"]
    data["metadata"]["version"] = 2
    with pytest.raises(ReportFormatError, match="inter_token_latency_ms"):
        parse_report(json.dumps(data), seed=1)
    with pytest.raises(ReportFormatError, match="JSON"):
        parse_report("not json", seed=1)


def install(executable: tuple[str, ...] = ("/usr/bin/guidellm",)) -> GuidellmInstall:
    return GuidellmInstall(version="0.8.1", runtime="process", executable=executable)


def target(**overrides) -> RunTarget:  # type: ignore[no-untyped-def]
    values = {"target": "http://127.0.0.1:8000", "model": "qwen3-0.6b",
              "tokenizer": "/cache/tok", "tokenizer_local": True}  # fmt: skip
    return RunTarget(**(values | overrides))


def test_argv_golden():
    argv = build_argv(
        install(), target(api_key=KEY), Workload(), seed=43, duration_s=30,
        report_path=Path("/work/report-43.json"),
    )  # fmt: skip
    assert argv == [
        "/usr/bin/guidellm", "run",
        "--backend", '{"kind":"openai_http","model":"qwen3-0.6b","target":"http://127.0.0.1:8000"}',
        "--tokenizer", '{"kind":"huggingface_auto","model":"/cache/tok"}',
        "--data", '{"kind":"synthetic_text","output_tokens":128,"prompt_tokens":256}',
        "--profile", '{"kind":"concurrent","streams":4}',
        "--constraint", '{"kind":"max_duration","seconds":30}',
        "--seed", '{"kind":"static","value":43}',
        "--metrics", '{"kind":"generative","sample_size":0}',
        "--output", '{"kind":"json","path":"/work/report-43.json"}',
        "--disable-console",
    ]  # fmt: skip
    assert KEY.value not in " ".join(argv)


def test_environment_carries_the_key_by_variable_only():
    base = {"PATH": "/bin", "BRASSKEY_API_KEY": "user-key", "GUIDELLM__SPEC__DATA": "x",
            "OPENAI_API_KEY": "other"}  # fmt: skip
    env = run_environment(target(api_key=KEY), base)
    assert env[KEY_ENV] == KEY.value
    assert "BRASSKEY_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "GUIDELLM__SPEC__DATA" not in env  # no stray GuideLLM settings from the shell
    assert env["HF_HUB_OFFLINE"] == "1"
    assert run_environment(target(tokenizer_local=False), base)["HF_HUB_OFFLINE"] == "0"


def runner_for(answers):  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def run(argv, *, timeout_s=5.0):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        for marker, answer in answers.items():
            if marker in argv:
                return answer
        return CommandOutput(1, "", "unexpected")

    return run, calls


def test_detect_on_path_then_pinned_uv():
    run, _ = runner_for({"--version": CommandOutput(0, "guidellm version: 0.8.1\n")})
    on_path = detect_guidellm(run=run, which={"guidellm": "/x/guidellm"}.get, fallback_dirs=())
    assert on_path is not None
    assert on_path.runtime == "process"
    ready = detect_guidellm(run=run, which={"uv": "/x/uv"}.get, fallback_dirs=())
    assert ready is not None
    assert ready.runtime == "uvx"
    assert not ready.needs_setup
    assert ready.executable == uv_base_command("/x/uv")
    assert "--torch-backend" in ready.executable  # CPU-only torch: no CUDA wheels


def test_pinned_uv_not_set_up_yet_needs_a_confirmed_setup():
    run, calls = runner_for({"--version": CommandOutput(2, "", "offline: not cached")})
    found = detect_guidellm(run=run, which={"uv": "/x/uv"}.get, fallback_dirs=())
    assert found is not None
    assert found.needs_setup
    assert "--offline" in calls[0]  # detection never downloads
    assert "--offline" not in found.setup_argv
    assert found.setup_note is not None
    assert detect_guidellm(run=run, which={}.get, fallback_dirs=()) is None


@pytest.fixture
def fake_install():
    return install((sys.executable, str(FAKE_GUIDELLM)))


def test_run_once_with_a_real_subprocess(fake_install, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_REQUIRE_KEY", KEY.value)
    work, log = tmp_path / "work", tmp_path / "bench.log"
    work.mkdir()
    metrics = run_once(fake_install, target(api_key=KEY), Workload(), seed=44, duration_s=5,
                       work_dir=work, log_path=log)  # fmt: skip
    assert metrics.output_tokens_per_s == pytest.approx(130.3966 * 1.02, abs=1e-3)
    assert not (work / "report-44.json").exists()  # the raw report is deleted at once
    recorded = json.loads((work / "report-44.json.argv").read_text())
    assert KEY.value not in json.dumps(recorded["argv"])
    assert Path(recorded["cwd"]).resolve() == work.resolve()  # runs in the private folder
    assert oct(log.stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("unauthorized", "refused GuideLLM's requests"),
        ("old-format", "report format 1"),
        ("no-report", "GuideLLM failed"),
        ("all-failed", "No request succeeded"),
    ],
)
def test_run_once_failures_are_plain(fake_install, tmp_path, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_GUIDELLM_MODE", mode)
    with pytest.raises(BenchError, match=message):
        run_once(fake_install, target(), Workload(), seed=1, duration_s=5, work_dir=tmp_path,
                 log_path=tmp_path / "log")  # fmt: skip
    assert not (tmp_path / "report-1.json").exists()


# --- endpoints -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "base", "root"),
    [
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("https://gpu.example/v1/models", "https://gpu.example/v1", "https://gpu.example"),
        ("http://h/proxy/v1", "http://h/proxy/v1", "http://h/proxy"),
    ],
)
def test_parse_endpoint(url, base, root):
    assert parse_endpoint(url) == Endpoint(api_base=base, root=root)


@pytest.mark.parametrize("url", ["ftp://x", "127.0.0.1:8000", "http://user:pw@h/v1"])
def test_bad_endpoints(url):
    with pytest.raises(BenchError):
        parse_endpoint(url)


def client_for(status: int, body: object = None) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.headers.get("authorization") not in (None, f"Bearer {KEY.value}"):
            return httpx2.Response(401)
        return httpx2.Response(status, json=body)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_served_models_and_errors():
    endpoint = parse_endpoint("http://h:1/v1")
    body = {"data": [{"id": "qwen3-0.6b"}]}
    assert served_models(client_for(200, body), endpoint, KEY) == ["qwen3-0.6b"]
    with pytest.raises(BenchError, match="needs an API key"):
        served_models(client_for(401), endpoint, None)
    with pytest.raises(BenchError, match="HTTP 500"):
        served_models(client_for(500), endpoint, None)

    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(BenchError, match="Nothing answered") as caught:
        served_models(httpx2.Client(transport=httpx2.MockTransport(down)), endpoint, None)
    assert caught.value.next_step.command == "brasskey doctor"


def test_choose_model():
    endpoint = parse_endpoint("http://h:1/v1")
    assert choose_model(endpoint, ["a"], None) == "a"
    assert choose_model(endpoint, ["a", "b"], "b") == "b"
    with pytest.raises(BenchError, match="several"):
        choose_model(endpoint, ["a", "b"], None)
    with pytest.raises(BenchError, match="doesn't serve"):
        choose_model(endpoint, ["a"], "c")


def test_brasskey_servers_are_recognized_by_url_or_loopback_port():
    from datetime import UTC, datetime

    from brasskey.runtime.state import ServerRecord

    record = ServerRecord(
        id="s", pid=1, process_started_at=1.0, engine="mlx", engine_version="0.32.2",
        model_id="qwen3-0.6b", host="127.0.0.1", port=8812, engine_port=40000,
        base_url="http://127.0.0.1:8812/v1", auth="none", started_at=datetime.now(UTC),
        log_path="x", brasskey_version="0",
    )  # fmt: skip
    assert brasskey_server(parse_endpoint("http://127.0.0.1:8812/v1"), [record]) is record
    assert brasskey_server(parse_endpoint("http://localhost:8812"), [record]) is record
    assert brasskey_server(parse_endpoint("http://127.0.0.1:9999"), [record]) is None


# --- tokenizers ------------------------------------------------------------------------------


def test_tokenizer_from_a_folder_or_an_id(tmp_path):
    hub = FakeHub(cache_dir=tmp_path / "hf")
    folder = choose_tokenizer("x", str(tmp_path), catalog=load_catalog(), hub=hub)
    assert folder.local
    by_id = choose_tokenizer("x", "Qwen/Qwen3-4B", catalog=load_catalog(), hub=hub)
    assert not by_id.local
    assert by_id.source == "Qwen/Qwen3-4B"


def test_catalog_tokenizer_downloads_only_the_tokenizer_files(tmp_path):
    hub = FakeHub(cache_dir=tmp_path / "hf")
    choice = choose_tokenizer("qwen3-0.6b", None, catalog=load_catalog(), hub=hub)
    assert choice.download is not None
    paths = {file.path for file in choice.download.files}
    assert "tokenizer.json" in paths
    assert not any(path.endswith((".safetensors", ".gguf")) for path in paths)
    assert choice.download_bytes < 20_000_000


def test_cached_tokenizer_files_are_used_offline(tmp_path):
    hub = FakeHub(cache_dir=tmp_path / "hf")
    model = next(m for m in load_catalog().models if m.id == "qwen3-0.6b")
    artifact = next(a for a in model.artifacts if a.format.value == "mlx")
    for file in artifact.files:
        hub.add_cached_blob(artifact.repo_id, file.path, b"x", file.sha256 or "0" * 64)
    choice = choose_tokenizer("qwen3-0.6b", None, catalog=load_catalog(), hub=hub)
    assert choice.download is None
    assert choice.local
    assert fetch_tokenizer(choice, hub) == choice


def test_unknown_models_need_tokenizer():
    hub = FakeHub(cache_dir=Path("/nonexistent"))
    with pytest.raises(BenchError) as caught:
        choose_tokenizer("hf:Qwen/Qwen3-4B-GGUF:Q8_0", None, catalog=load_catalog(), hub=hub)
    assert (caught.value.fix or "").endswith("--tokenizer Qwen/Qwen3-4B")
    with pytest.raises(BenchError, match="doesn't know which tokenizer") as unknown:
        choose_tokenizer("my-model", None, catalog=load_catalog(), hub=hub)
    assert "--tokenizer" in (unknown.value.fix or "")


# --- run order and analysis ------------------------------------------------------------------


def test_schedule_warms_up_then_alternates():
    steps = schedule(Workload(repeats=3, warmup_s=10, seed=42), ["candidate", "baseline"])
    assert [(s.role, s.seed, s.measured) for s in steps] == [
        ("candidate", 1042, False), ("baseline", 1042, False),
        ("candidate", 42, True), ("baseline", 42, True),
        ("baseline", 43, True), ("candidate", 43, True),
        ("candidate", 44, True), ("baseline", 44, True),
    ]  # fmt: skip
    assert estimate_seconds(steps) == 2 * 10 + 6 * 30 + 8 * 12
    assert schedule(Workload(warmup_s=0, repeats=1), ["candidate"])[0].measured


def run_metrics(seed: int, throughput: float) -> RunMetrics:
    return RunMetrics(
        seed=seed, duration_s=30, successful=100, errored=0, incomplete=0,
        output_tokens_per_s=throughput, requests_per_s=throughput / 30, ttft_ms_p50=300,
        ttft_ms_p95=400 + seed, itl_ms_p50=6.5, latency_s_p50=0.5, latency_s_p95=0.6 + seed / 1000,
    )  # fmt: skip


def test_execute_discards_warmups():
    steps = schedule(Workload(repeats=2), ["candidate"])
    results = execute(steps, lambda step: run_metrics(step.seed, 100.0))
    assert [run.seed for run in results["candidate"]] == [42, 43]


def test_summary_and_comparison_decided_by_the_primary_metric():
    base_runs = [run_metrics(i, value) for i, value in enumerate((100, 101, 99))]
    cand_runs = [run_metrics(i, value) for i, value in enumerate((130, 128, 131))]

    def result(runs):  # type: ignore[no-untyped-def]
        return EndpointResult(url="u", model="m", runs=tuple(runs), metrics=summarize(runs))

    summary = result(cand_runs).metric("output_tokens_per_s")
    assert summary.mean == pytest.approx(129.6667, abs=1e-3)
    assert not summary.noisy
    outcome = compare(result(base_runs), result(cand_runs), goal="better", margin=0.0,
                      primary="output_tokens_per_s", baseline_source="endpoint")  # fmt: skip
    assert outcome.verdict == Verdict.PASS
    ttft = next(item for item in outcome.metrics if item.key == "ttft_ms_p50")
    assert ttft.verdict == Verdict.INCONCLUSIVE  # identical in every run: too coarse
    assert outcome.reason.startswith("Output tokens/s")
