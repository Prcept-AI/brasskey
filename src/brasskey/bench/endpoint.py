# SPDX-License-Identifier: Apache-2.0
"""The endpoints a benchmark talks to: URL checks, the served model, and Brasskey servers."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import httpx2

from brasskey.bench.guidellm import BenchError
from brasskey.output.next_step import NextStep
from brasskey.runtime.state import ServerRecord
from brasskey.security.apikeys import ApiKey

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Endpoint:
    api_base: str  # e.g. http://127.0.0.1:8000/v1
    root: str  # e.g. http://127.0.0.1:8000 (what GuideLLM calls the target)

    @property
    def is_https(self) -> bool:
        return self.api_base.startswith("https://")


def parse_endpoint(url: str) -> Endpoint:
    """Accept ``http://host:port`` or ``http://host:port/v1`` (an OpenAI-compatible base URL)."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BenchError(
            f"{url!r} isn't an http(s) URL.",
            fix="Pass the server's base URL, e.g. http://127.0.0.1:8000/v1",
            code="BK-E-USAGE",
        )
    if parts.username or parts.password:
        raise BenchError(
            "The URL contains credentials.",
            fix="Remove them from the URL and put the API key in an environment variable "
            "(BRASSKEY_API_KEY, or name one with --api-key-env).",
            code="BK-E-USAGE",
        )
    path = parts.path.rstrip("/")
    if path.endswith("/models"):
        path = path.removesuffix("/models")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    root = f"{parts.scheme}://{parts.netloc}{path.removesuffix('/v1')}"
    return Endpoint(api_base=f"{root}/v1", root=root)


def served_models(client: httpx2.Client, endpoint: Endpoint, key: ApiKey | None) -> list[str]:
    """Model ids the endpoint lists; plain errors for the usual problems."""
    headers = {"Authorization": f"Bearer {key.value}"} if key is not None else {}
    try:
        response = client.get(f"{endpoint.api_base}/models", headers=headers)
    except httpx2.HTTPError as error:
        raise BenchError(
            f"Nothing answered at {endpoint.api_base}.",
            fix="Start a server first (`brasskey run`), or check the URL and port.",
            code="BK-E-BENCH-ENDPOINT",
            next_step=NextStep(command="brasskey doctor", hint="see the running servers"),
        ) from error
    if response.status_code in (401, 403):
        raise BenchError(
            f"{endpoint.api_base} needs an API key"
            + (" and refused this one." if key is not None else "."),
            fix="Set BRASSKEY_API_KEY (or name the variable with --api-key-env), then run again.",
            code="BK-E-BENCH-AUTH",
        )
    if response.status_code != 200:
        raise BenchError(
            f"{endpoint.api_base}/models answered HTTP {response.status_code}.",
            fix="Check that this is an OpenAI-compatible server.",
            code="BK-E-BENCH-ENDPOINT",
        )
    try:
        body: object = response.json()
    except ValueError as error:
        raise BenchError(f"{endpoint.api_base}/models didn't return JSON.") from error
    entries = cast(dict[str, object], body).get("data") if isinstance(body, dict) else None
    ids = [
        str(cast(dict[str, object], entry).get("id"))
        for entry in (cast(list[object], entries) if isinstance(entries, list) else [])
        if isinstance(entry, dict) and cast(dict[str, object], entry).get("id")
    ]
    logger.debug("%s serves %d model(s)", endpoint.api_base, len(ids))
    return ids


def choose_model(endpoint: Endpoint, served: Sequence[str], requested: str | None) -> str:
    if requested is not None:
        if served and requested not in served:
            raise BenchError(
                f"{endpoint.api_base} doesn't serve {requested!r} (it serves {', '.join(served)}).",
                fix="Pass one of those with --model.",
                code="BK-E-USAGE",
            )
        return requested
    if len(served) == 1:
        return served[0]
    if not served:
        raise BenchError(
            f"{endpoint.api_base} lists no models.",
            fix="Pass the model name with --model.",
            code="BK-E-USAGE",
        )
    raise BenchError(
        f"{endpoint.api_base} serves several models: {', '.join(served)}.",
        fix="Pick one with --model.",
        code="BK-E-USAGE",
    )


def brasskey_server(endpoint: Endpoint, records: Sequence[ServerRecord]) -> ServerRecord | None:
    """The running Brasskey server behind this URL, if any (to label results with its engine)."""
    for record in records:
        if record.base_url.rstrip("/") == endpoint.api_base:
            return record
        loopback = {"127.0.0.1", "localhost", "[::1]"}
        mine, theirs = urlsplit(endpoint.api_base), urlsplit(record.base_url)
        if (
            mine.port == theirs.port
            and mine.hostname in loopback | {"::1"}
            and theirs.hostname in loopback | {"::1"}
        ):
            return record
    return None
