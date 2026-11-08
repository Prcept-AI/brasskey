# SPDX-License-Identifier: Apache-2.0
"""``doctor --security``: checks over fake machine states, parsers, probes and the CLI."""

import ipaddress
from datetime import UTC, datetime
from typing import Literal

import httpx2
import pytest

from brasskey.runtime.state import ServerRecord
from brasskey.security.apikeys import ApiKey
from brasskey.security.doctor import (
    CheckStatus,
    SecurityContext,
    check_url,
    model_server_candidates,
    models_url,
    run_checks,
)
from brasskey.security.netinfo import (
    PROBE_ORIGIN,
    DockerPublish,
    HttpProbe,
    Listener,
    ProcessInfo,
    make_prober,
    parse_docker_ps,
    parse_lsof,
)
from fakes.helpers import fake_api_key

PRIVATE = ipaddress.ip_address("192.168.1.20")


def record(
    host: str = "127.0.0.1",
    port: int = 8000,
    auth: Literal["none", "api_key"] = "none",
    tls: bool = False,
) -> ServerRecord:
    return ServerRecord(
        id="srv", pid=1, process_started_at=1.0, engine="llamacpp", engine_version="1",
        model_id="qwen3-4b", host=host, port=port, engine_port=40000, tls=tls,
        base_url=f"http://{host}:{port}/v1", auth=auth, started_at=datetime.now(UTC),
        log_path="server.log", brasskey_version="0",
    )  # fmt: skip


def open_probe(*, cors: str | None = None):
    def probe(url):
        return HttpProbe(url=url, status=200, allow_origin=cors, looks_like_models=True)

    return probe


def closed_probe(url):
    return HttpProbe(url=url, status=401, allow_origin=None, looks_like_models=False)


def context(**overrides) -> SecurityContext:
    values = {
        "servers": [], "listeners": [], "addresses": [PRIVATE], "docker": [], "firewall": "on",
        "processes": [], "api_key": None, "probe": closed_probe, "os_family": "darwin",
    }  # fmt: skip
    return SecurityContext(**(values | overrides))


def by_id(results, check_id):
    return [result for result in results if result.id == check_id]


def statuses(results, check_id):
    return [result.status for result in by_id(results, check_id)]


def test_a_quiet_safe_machine():
    results = run_checks(context())
    assert not [r for r in results if r.status in (CheckStatus.FAIL, CheckStatus.WARN)]
    assert statuses(results, "BK-SEC-009") == [CheckStatus.PASS]


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (record(), CheckStatus.PASS),
        (record(host="0.0.0.0", auth="api_key", tls=True), CheckStatus.PASS),  # noqa: S104
        (record(host="0.0.0.0", auth="api_key"), CheckStatus.WARN),  # noqa: S104
        (record(host="0.0.0.0"), CheckStatus.FAIL),  # noqa: S104
    ],
)
def test_brasskey_servers(server, expected):
    assert statuses(run_checks(context(servers=[server])), "BK-SEC-001") == [expected]


def test_ollama_on_all_interfaces_without_a_key_fails_with_a_specific_fix():
    ollama = Listener("*", 11434, 501, "ollama")
    results = run_checks(context(listeners=[ollama], probe=open_probe()))
    (finding,) = by_id(results, "BK-SEC-002")
    assert finding.status == CheckStatus.FAIL
    assert "OLLAMA_HOST=127.0.0.1" in (finding.fix or "")


def test_open_server_on_loopback_is_acceptable():
    ollama = Listener("127.0.0.1", 11434, 501, "ollama")
    results = run_checks(context(listeners=[ollama], probe=open_probe()))
    assert statuses(results, "BK-SEC-002") == [CheckStatus.PASS]


def test_any_origin_lets_web_pages_read_answers():
    mlx = Listener("127.0.0.1", 8080, 77, "Python mlx_lm.server")
    results = run_checks(context(listeners=[mlx], probe=open_probe(cors="*")))
    (finding,) = by_id(results, "BK-SEC-003")
    assert finding.status == CheckStatus.WARN
    assert "--allowed-origins" in (finding.fix or "")


def test_familiar_port_numbers_alone_are_not_reported():
    airplay = Listener("*", 5000, 600, "ControlCe")  # macOS AirPlay Receiver on port 5000
    results = run_checks(context(listeners=[airplay], probe=closed_probe))
    assert statuses(results, "BK-SEC-002") == [CheckStatus.SKIP]


def test_brasskey_engine_ports_are_not_probed():
    engine = Listener("127.0.0.1", 40000, 9, "llama-server")
    assert model_server_candidates(context(servers=[record()], listeners=[engine])) == []


def test_public_address_with_wide_listeners_fails():
    ollama = Listener("0.0.0.0", 11434, 501, "ollama")  # noqa: S104
    public = ipaddress.ip_address("8.8.8.8")
    results = run_checks(context(listeners=[ollama], addresses=[public], probe=closed_probe))
    assert statuses(results, "BK-SEC-004") == [CheckStatus.FAIL]
    quiet = run_checks(context(addresses=[public]))
    assert statuses(quiet, "BK-SEC-004") == [CheckStatus.PASS]


def test_docker_publishing_model_ports_on_all_interfaces():
    vllm = DockerPublish("vllm-1", "vllm/vllm-openai:v0.11", "0.0.0.0", 8000, 8000)  # noqa: S104
    safe = DockerPublish("vllm-2", "vllm/vllm-openai:v0.11", "127.0.0.1", 8001, 8000)
    results = run_checks(context(docker=[vllm, safe]))
    (finding,) = by_id(results, "BK-SEC-005")
    assert finding.status == CheckStatus.WARN
    assert "-p 127.0.0.1:8000:8000" in (finding.fix or "")
    assert statuses(run_checks(context(docker=None)), "BK-SEC-005") == [CheckStatus.SKIP]


@pytest.mark.parametrize(
    ("firewall", "os_family", "expected"),
    [("on", "darwin", CheckStatus.PASS), ("off", "darwin", CheckStatus.WARN),
     (None, "linux", CheckStatus.SKIP)],
)  # fmt: skip
def test_firewall(firewall, os_family, expected):
    results = run_checks(context(firewall=firewall, os_family=os_family))
    assert statuses(results, "BK-SEC-006") == [expected]


def test_api_keys_on_command_lines_are_flagged_without_showing_them():
    leaky = ProcessInfo(4242, "llama-server", ("llama-server", "--api-key", "sk-live-secret"))
    fine = ProcessInfo(4243, "brasskey", ("brasskey", "run", "--api-key-env", "TEAM_KEY"))
    results = run_checks(context(processes=[leaky, fine]))
    (finding,) = by_id(results, "BK-SEC-007")
    assert finding.status == CheckStatus.WARN
    assert "4242" in finding.detail
    assert "sk-live-secret" not in finding.detail


@pytest.mark.parametrize(
    ("argv", "leaks"),
    [
        (("docker", "run", "-e", "VLLM_API_KEY=sk-live-123", "img"), True),
        (("docker", "run", "--env=HF_TOKEN=hf_abc", "img"), True),
        (("docker", "run", "--env", "OPENAI_API_KEY=sk-1", "img"), True),
        (("docker", "run", "-e", "VLLM_API_KEY", "-e", "DO_NOT_TRACK=1", "img"), False),
        (("docker", "run", "-e", "HF_TOKEN=", "img"), False),
    ],
)
def test_docker_env_flags_with_secret_values_are_flagged(argv, leaks):
    results = run_checks(context(processes=[ProcessInfo(99, "docker", argv)]))
    expected = CheckStatus.WARN if leaks else CheckStatus.PASS
    assert statuses(results, "BK-SEC-007") == [expected]


def test_api_key_strength():
    weak = run_checks(context(api_key=ApiKey("short-key-1", "env:BRASSKEY_API_KEY")))
    assert statuses(weak, "BK-SEC-008") == [CheckStatus.WARN]
    strong = run_checks(context(api_key=ApiKey(fake_api_key("strong"), "keyring")))
    assert statuses(strong, "BK-SEC-008") == [CheckStatus.PASS]


# --- endpoint checks -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://gpu-box:8000/v1", "http://gpu-box:8000/v1/models"),
        ("http://gpu-box:8000", "http://gpu-box:8000/v1/models"),
        ("https://gpu-box/v1/models/", "https://gpu-box/v1/models"),
    ],
)
def test_models_url(url, expected):
    assert models_url(url) == expected


def test_plain_http_to_another_machine_fails():
    results = check_url("http://gpu-box:8000/v1", closed_probe)
    assert statuses(results, "BK-SEC-010") == [CheckStatus.FAIL]
    assert statuses(results, "BK-SEC-002") == [CheckStatus.PASS]


def test_open_endpoint_on_the_network_fails_and_cors_warns():
    results = check_url("https://gpu-box/v1", open_probe(cors="*"))
    assert statuses(results, "BK-SEC-010") == [CheckStatus.PASS]
    assert statuses(results, "BK-SEC-002") == [CheckStatus.FAIL]
    assert statuses(results, "BK-SEC-003") == [CheckStatus.WARN]


def test_https_that_cannot_connect_fails():
    assert statuses(check_url("https://gpu-box/v1", lambda _url: None), "BK-SEC-010") == [
        CheckStatus.FAIL
    ]


# --- parsers and the probe -----------------------------------------------------------------------


def test_parse_lsof_field_output():
    text = "p501\ncollama\nf3\nn*:11434\np777\ncPython\nf5\nn127.0.0.1:8080\nf6\nn[::1]:8080\n"
    assert parse_lsof(text) == [
        Listener("*", 11434, 501, "ollama"),
        Listener("127.0.0.1", 8080, 777, "Python"),
        Listener("[::1]", 8080, 777, "Python"),
    ]


def test_parse_docker_ps():
    line = ('{"Names":"vllm-1","Image":"vllm/vllm-openai","Ports":"0.0.0.0:8000->8000/tcp, '
            ':::8000->8000/tcp"}')  # fmt: skip
    published = parse_docker_ps(line + "\nnot json\n")
    found = [(p.host_address, p.host_port) for p in published]
    assert found == [("0.0.0.0", 8000), ("::", 8000)]  # noqa: S104


def test_prober_sends_a_foreign_origin_and_reads_cors():
    seen = {}

    def handler(request):
        seen["origin"] = request.headers.get("origin")
        return httpx2.Response(200, json={"data": []}, headers={"access-control-allow-origin": "*"})

    probe = make_prober(httpx2.Client(transport=httpx2.MockTransport(handler)))(
        "http://x/v1/models"
    )
    assert seen["origin"] == PROBE_ORIGIN
    assert probe is not None
    assert probe.open_without_key
    assert probe.readable_by_any_site


# --- the command ---------------------------------------------------------------------------------


def test_doctor_security_exit_code_and_json(cli, monkeypatch):
    risky = context(listeners=[Listener("*", 11434, 501, "ollama")], probe=open_probe())
    monkeypatch.setattr("brasskey.cli.doctor.gather_security_context", lambda _family: risky)
    result = cli("doctor", "--security")
    assert result.exit_code == 3
    assert "FAIL  BK-SEC-002" in result.stdout
    assert "Fix: Set OLLAMA_HOST=127.0.0.1" in result.stdout
    assert result.last_line().startswith("Next: brasskey doctor --security")
    document = cli("doctor", "--security", "--json").json()
    assert document["ok"] is True  # the command worked; the verdict is in data and the exit code
    assert document["data"]["passed"] is False
    assert document["data"]["summary"]["fail"] == 1


def test_doctor_security_passes(cli, monkeypatch):
    monkeypatch.setattr("brasskey.cli.doctor.gather_security_context", lambda _family: context())
    result = cli("doctor", "--security")
    assert result.exit_code == 0
    assert result.last_line() == "Next: brasskey run (start a model)"


def test_doctor_security_for_a_url(cli, monkeypatch):
    monkeypatch.setattr("brasskey.cli.doctor.make_url_prober", lambda: closed_probe)
    result = cli("doctor", "--security", "http://gpu-box:8000/v1")
    assert result.exit_code == 3
    assert "Security checks for http://gpu-box:8000/v1" in result.stdout


def test_a_url_without_security_is_a_usage_error(cli):
    result = cli("doctor", "http://gpu-box:8000/v1")
    assert result.exit_code == 2
    assert "doctor --security http://gpu-box:8000/v1" in result.stderr
