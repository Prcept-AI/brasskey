# SPDX-License-Identifier: Apache-2.0
"""``brasskey.yaml``: the intent. Only fields Brasskey acts on today; new ones are added later.

Example::

    version: 1
    model: qwen3-4b          # catalog id or hf:org/name[:QUANT]; omit to let Brasskey pick
    engine: llamacpp         # optional
    goal:
      context_length: 8192
      concurrency: 1
    serve:
      host: 127.0.0.1        # anything else needs an API key (BRASSKEY_API_KEY or the keyring)
      port: 8000             # optional; default: 8000 or the next free port
      tls:                   # optional HTTPS
        cert_file: /path/cert.pem
        key_file: /path/key.pem
"""

import ipaddress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from brasskey.engines.types import EngineId
from brasskey.models.fit import MIN_CONTEXT

INTENT_FILE = "brasskey.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class GoalIntent(_Strict):
    context_length: int = Field(
        default=8192, ge=MIN_CONTEXT, description="Tokens of context to plan memory for."
    )
    concurrency: int = Field(default=1, ge=1, description="Sequences served at the same time.")


class TlsIntent(_Strict):
    cert_file: Path = Field(description="PEM certificate (chain) file.")
    key_file: Path = Field(description="PEM private key file.")


class ServeIntent(_Strict):
    host: str = Field(
        default="127.0.0.1",
        description="Address to listen on. Anything but loopback requires an API key.",
    )
    port: int | None = Field(
        default=None, ge=1, le=65535, description="Port; default 8000 or the next free one."
    )
    api_key_env: str | None = Field(
        default=None,
        pattern=r"^[A-Z_][A-Z0-9_]*$",
        description="Name of the environment variable holding the API key (never the key itself).",
    )
    tls: TlsIntent | None = Field(default=None, description="Serve HTTPS with these files.")


class Intent(_Strict):
    version: Literal[1] = 1
    model: str | None = Field(
        default=None, description="Catalog id or hf:org/name[:QUANT]; omit to let Brasskey pick."
    )
    engine: EngineId | None = Field(default=None, description="Engine to use; omit to let it pick.")
    goal: GoalIntent = GoalIntent()
    serve: ServeIntent = ServeIntent()
