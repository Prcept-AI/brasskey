# SPDX-License-Identifier: Apache-2.0
"""The Brasskey proxy: one front door with the same security for every engine (ADR 0008, 0009).

Clients talk to the proxy at the address the user asked for; the engine listens only on a private
loopback port. For every request the proxy:

1. refuses browser requests from other web origins (403), so a web page you visit can't use or
   read the server (engines like mlx-lm allow any origin by default);
2. requires the API key when one is configured (401), compared in constant time;
3. answers ``/v1/models`` itself (only the served model) and ``/health`` (no key needed), and
   passes ``/metrics`` through (same checks) when the engine has one;
4. refuses requests that name another model (404), so a client can't make an engine load
   something outside the memory plan;
5. caps request bodies (413), then streams everything under ``/v1/`` to the engine and the
   engine's response back unchanged (including server-sent events).

It never logs request or response bodies, headers or keys. The engine's own key (llama.cpp) is a
random per-run secret only the proxy knows, so local programs can't bypass the proxy either.
"""

import contextlib
import hmac
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx2
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from brasskey.errors import BrasskeyError
from brasskey.hardware.types import MIB

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 32 * MIB
HOP_BY_HOP = frozenset(
    {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
        "trailer", "transfer-encoding", "upgrade", "content-length", "host",
    }
)  # fmt: skip
# Never passed to the engine: the client's credentials, cookies and browser origin.
CLIENT_ONLY = frozenset({"authorization", "x-api-key", "cookie", "origin", "referer"})


class ProxyStartError(BrasskeyError):
    default_code = "BK-E-PROXY"


@dataclass(frozen=True)
class ProxyConfig:
    upstream: str  # e.g. http://127.0.0.1:52311 (the engine's private address)
    served_model: str
    api_key: str | None = field(default=None, repr=False)
    upstream_api_key: str | None = field(default=None, repr=False)
    allowed_origins: frozenset[str] = frozenset()
    max_body_bytes: int = MAX_BODY_BYTES


def _error(status: int, message: str, code: str) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind, "code": code}}, status)


def _presented_key(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return request.headers.get("x-api-key")


async def _read_limited(request: Request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than ``limit``."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(config: ProxyConfig) -> Starlette:
    clients: dict[str, httpx2.AsyncClient] = {}

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
        clients["engine"] = httpx2.AsyncClient(
            timeout=httpx2.Timeout(connect=5.0, read=None, write=60.0, pool=10.0),
            trust_env=False,
        )
        try:
            yield
        finally:
            await clients["engine"].aclose()

    def guard(request: Request) -> Response | None:
        origin = request.headers.get("origin")
        if origin is not None and origin not in config.allowed_origins:
            return _error(
                403,
                "Requests from web pages on other sites are not allowed.",
                "origin_not_allowed",
            )
        if config.api_key is not None:
            presented = _presented_key(request)
            if presented is None or not hmac.compare_digest(
                presented.encode(), config.api_key.encode()
            ):
                return _error(
                    401,
                    "Missing or wrong API key. Send it as 'Authorization: Bearer <key>'.",
                    "invalid_api_key",
                )
        return None

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "model": config.served_model})

    async def models(request: Request) -> Response:
        denied = guard(request)
        if denied is not None:
            return denied
        entry = {"id": config.served_model, "object": "model", "created": 0, "owned_by": "brasskey"}
        return JSONResponse({"object": "list", "data": [entry]})

    async def forward(request: Request) -> Response:
        denied = guard(request)
        if denied is not None:
            return denied
        body = await _read_limited(request, config.max_body_bytes)
        if body is None:
            return _error(413, "The request body is too large.", "request_too_large")
        if body and "json" in request.headers.get("content-type", ""):
            try:
                payload: object = json.loads(body)
            except ValueError:
                payload = None  # let the engine report malformed JSON its own way
            if isinstance(payload, dict):
                fields = cast(dict[str, object], payload)
                requested = fields.get("model")
                if requested is not None and requested != config.served_model:
                    return _error(
                        404,
                        f"This server serves '{config.served_model}', not '{requested}'.",
                        "model_not_found",
                    )
                if requested is None:
                    fields["model"] = config.served_model
                    body = json.dumps(fields).encode()
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in HOP_BY_HOP and name.lower() not in CLIENT_ONLY
        }
        if config.upstream_api_key:
            headers["authorization"] = f"Bearer {config.upstream_api_key}"
        url = config.upstream + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        client = clients["engine"]
        try:
            upstream = await client.send(
                client.build_request(request.method, url, headers=headers, content=body),
                stream=True,
            )
        except httpx2.HTTPError:
            logger.debug("Engine unreachable for %s %s", request.method, request.url.path)
            return _error(502, "The engine isn't responding.", "engine_unavailable")
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in HOP_BY_HOP
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/metrics", forward, methods=["GET"]),  # Prometheus counters, behind the same checks
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/models/{rest:path}", models, methods=["GET"]),
        Route("/v1/{rest:path}", forward, methods=["GET", "POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


class ProxyServer:
    """Runs the proxy (uvicorn) in a background thread; Brasskey's main thread keeps signals."""

    def __init__(
        self,
        app: Starlette,
        *,
        host: str,
        port: int,
        tls_cert: Path | None = None,
        tls_key: Path | None = None,
    ) -> None:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.handlers = [logging.NullHandler()]
            uvicorn_logger.propagate = False
        self.host = host
        self.port = port
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_config=None,
            log_level="warning",
            access_log=False,  # no request lines (paths can carry identifiers)
            server_header=False,
            proxy_headers=False,
            ssl_certfile=str(tls_cert) if tls_cert else None,
            ssl_keyfile=str(tls_key) if tls_key else None,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._serve, name="brasskey-proxy", daemon=True)

    def _serve(self) -> None:
        try:
            self._server.run()
        except SystemExit:  # uvicorn exits this way when it can't bind; start() reports it
            logger.debug("Proxy failed to start on %s:%d", self.host, self.port)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self, timeout_s: float = 15.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise ProxyStartError(
                    f"Brasskey couldn't listen on {self.host}:{self.port}.",
                    fix="Pick another --port or --host (the address must belong to this machine).",
                )
            time.sleep(0.02)
        logger.debug("Proxy listening on %s:%d", self.host, self.port)

    def stop(self, timeout_s: float = 10.0) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout_s)
