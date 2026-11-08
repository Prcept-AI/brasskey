# SPDX-License-Identifier: Apache-2.0
"""The Brasskey proxy, for real: uvicorn in front of the fake engine process."""

import json
import subprocess
import sys
import time

import httpx2
import pytest

from brasskey.runtime.proxy import ProxyConfig, ProxyServer, ProxyStartError, create_app
from fakes.engine import FAKE_ENGINE
from fakes.helpers import fake_api_key, free_port

KEY = fake_api_key("client")
ENGINE_KEY = fake_api_key("engine")


@pytest.fixture(scope="module")
def engine():
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(FAKE_ENGINE), "--port", str(port), "--model-name", "tiny",
         "--require-key", ENGINE_KEY, "--load-seconds", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx2.get(f"http://127.0.0.1:{port}/health", trust_env=False).status_code == 200:
                break
        except httpx2.HTTPError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    process.terminate()
    process.wait(5)


def start_proxy(engine_url, **overrides):
    port = free_port()
    config = ProxyConfig(
        upstream=engine_url,
        served_model="tiny",
        api_key=KEY,
        upstream_api_key=ENGINE_KEY,
        allowed_origins=frozenset({f"http://127.0.0.1:{port}"}),
    )
    config = ProxyConfig(**{**config.__dict__, **overrides})
    server = ProxyServer(create_app(config), host="127.0.0.1", port=port)
    server.start()
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture
def proxy(engine):
    server, url = start_proxy(engine)
    yield url
    server.stop()


@pytest.fixture
def http():
    with httpx2.Client(timeout=10, trust_env=False) as client:
        yield client


AUTH = {"Authorization": f"Bearer {KEY}"}
CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}


def test_requires_the_client_key(proxy, http):
    assert http.get(f"{proxy}/v1/models").status_code == 401
    wrong = http.get(f"{proxy}/v1/models", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_api_key"
    assert http.get(f"{proxy}/v1/models", headers={"x-api-key": KEY}).status_code == 200


def test_lists_only_the_served_model(proxy, http):
    models = http.get(f"{proxy}/v1/models", headers=AUTH).json()
    assert [entry["id"] for entry in models["data"]] == ["tiny"]


def test_health_needs_no_key(proxy, http):
    assert http.get(f"{proxy}/health").json() == {"status": "ok", "model": "tiny"}


def test_forwards_with_the_engine_key_and_strips_client_credentials(proxy, http):
    reply = http.post(f"{proxy}/v1/chat/completions", headers={**AUTH, "Cookie": "a=b"}, json=CHAT)
    assert reply.status_code == 200  # the fake engine demanded ENGINE_KEY: the proxy sent it
    assert reply.json()["saw_client_credentials"] is False


def test_fills_in_a_missing_model_name(proxy, http):
    reply = http.post(f"{proxy}/v1/chat/completions", headers=AUTH, json={"messages": []}).json()
    assert reply["model"] == "tiny"


def test_refuses_other_models(proxy, http):
    reply = http.post(f"{proxy}/v1/chat/completions", headers=AUTH, json={**CHAT, "model": "big"})
    assert reply.status_code == 404
    assert reply.json()["error"]["code"] == "model_not_found"


def test_refuses_other_web_origins_and_allows_its_own(proxy, http):
    evil = http.post(f"{proxy}/v1/chat/completions",
                     headers={**AUTH, "Origin": "https://evil.example"}, json=CHAT)  # fmt: skip
    assert evil.status_code == 403
    assert evil.json()["error"]["code"] == "origin_not_allowed"
    own = http.post(f"{proxy}/v1/chat/completions", headers={**AUTH, "Origin": proxy}, json=CHAT)
    assert own.status_code == 200


def test_streams_server_sent_events(proxy, http):
    events = []
    with http.stream("POST", f"{proxy}/v1/chat/completions", headers=AUTH,
                     json={**CHAT, "stream": True}) as response:  # fmt: skip
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:])["choices"][0]["delta"]["content"])
    assert "".join(events) == "Hello from the fake engine."


def test_metrics_pass_through_behind_the_key(proxy, http):
    assert http.get(f"{proxy}/metrics").status_code == 401
    assert "fake_requests_total" in http.get(f"{proxy}/metrics", headers=AUTH).text


def test_large_bodies_are_refused(engine, http):
    server, url = start_proxy(engine, max_body_bytes=100)
    try:
        reply = http.post(f"{url}/v1/chat/completions", headers=AUTH,
                          json={**CHAT, "padding": "x" * 500})  # fmt: skip
        assert reply.status_code == 413
    finally:
        server.stop()


def test_without_a_key_configured_anyone_local_may_call(engine, http):
    server, url = start_proxy(engine, api_key=None)
    try:
        assert http.get(f"{url}/v1/models").status_code == 200
    finally:
        server.stop()


def test_engine_down_is_a_502(http):
    server, url = start_proxy(f"http://127.0.0.1:{free_port()}")
    try:
        reply = http.post(f"{url}/v1/chat/completions", headers=AUTH, json=CHAT)
        assert reply.status_code == 502
        assert reply.json()["error"]["code"] == "engine_unavailable"
    finally:
        server.stop()


def test_a_busy_port_is_a_plain_error(proxy):
    port = int(proxy.rsplit(":", 1)[1])
    app = create_app(ProxyConfig(upstream="http://127.0.0.1:1", served_model="tiny"))
    with pytest.raises(ProxyStartError, match="couldn't listen"):
        ProxyServer(app, host="127.0.0.1", port=port).start(timeout_s=5)
