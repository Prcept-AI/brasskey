# SPDX-License-Identifier: Apache-2.0
"""How a server may be exposed: address, API key and TLS (ADR 0009). Pure decisions, no I/O
beyond reading the environment and the keyring through injected functions.

- Loopback (127.0.0.1, ::1, localhost): an API key is optional; a configured key is enforced.
- Anything else (a LAN address, 0.0.0.0, ::): an API key is required and must be strong. Without
  TLS, Brasskey warns that prompts, answers and the key travel unencrypted.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from brasskey.config.intent import is_loopback
from brasskey.security.apikeys import (
    DEFAULT_KEY_ENV,
    ApiKey,
    ApiKeyError,
    KeyringGet,
    KeyringSet,
    default_keyring_get,
    default_keyring_set,
    generate_api_key,
    key_weakness,
    resolve_api_key,
    store_api_key,
)

KEY_REQUIRED_CODE = "BK-E-API-KEY-REQUIRED"  # a network address without any key
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})  # noqa: S104 - recognised, not bound here


@dataclass(frozen=True)
class ServeSecurity:
    host: str
    api_key: ApiKey | None
    tls_cert: Path | None
    tls_key: Path | None
    generated: bool = False  # --generate-api-key created api_key just now (show it once)
    stored_in_keyring: bool = False
    warnings: tuple[str, ...] = ()
    key_env: str = DEFAULT_KEY_ENV  # the variable a client key is read from on later runs

    @property
    def loopback(self) -> bool:
        return is_loopback(self.host)

    @property
    def wildcard(self) -> bool:
        return self.host in WILDCARD_HOSTS

    @property
    def scheme(self) -> str:
        return "https" if self.tls_cert else "http"

    @property
    def client_host(self) -> str:
        """The address a client on this machine uses."""
        if self.host in {"0.0.0.0", "localhost"}:  # noqa: S104
            return "127.0.0.1"
        if self.host in {"::", "[::]"}:
            return "[::1]"
        return f"[{self.host}]" if ":" in self.host else self.host

    def origins(self, port: int) -> frozenset[str]:
        """Browser origins allowed to call the server: only its own addresses."""
        hosts = {"127.0.0.1", "localhost", "[::1]", self.client_host}
        if not self.wildcard:
            hosts.add(f"[{self.host}]" if ":" in self.host else self.host)
        return frozenset(f"{self.scheme}://{host}:{port}" for host in hosts)

    @property
    def access_label(self) -> str:
        if self.api_key is None:
            return "no API key (this machine only)"
        source = self.api_key.source
        if source == "keyring":
            where = "from your keychain"
        elif source.startswith("env:"):
            where = f"from {source.removeprefix('env:')}"
        else:  # "generated": created just now and not stored anywhere
            where = "generated for this run"
        return f"API key required ({where})"


def decide_security(
    host: str,
    *,
    api_key_env: str | None,
    generate: bool,
    tls_cert: Path | None,
    tls_key: Path | None,
    env: Mapping[str, str] | None = None,
    keyring_get: KeyringGet = default_keyring_get,
    keyring_set: KeyringSet = default_keyring_set,
    new_key: Callable[[], str] = generate_api_key,
) -> ServeSecurity:
    """Check the address, key and TLS settings; raise a plain error when they aren't safe."""
    source = os.environ if env is None else env
    if (tls_cert is None) != (tls_key is None):
        raise ApiKeyError(
            "TLS needs both a certificate and a private key.",
            fix="Pass both --tls-cert and --tls-key (or set both in brasskey.yaml).",
            code="BK-E-TLS",
        )
    for path in (tls_cert, tls_key):
        if path is not None and not path.is_file():
            raise ApiKeyError(f"TLS file not found: {path}.", code="BK-E-TLS")

    stored = False
    if generate:
        value = new_key()
        # An explicitly named variable never falls back to the keychain (resolve_api_key), so a
        # key stored there would be invisible next time: the user exports the variable instead.
        if api_key_env is None:
            stored = store_api_key(value, keyring_set=keyring_set)
        api_key: ApiKey | None = ApiKey(value=value, source="keyring" if stored else "generated")
    else:
        api_key = resolve_api_key(api_key_env, env=source, keyring_get=keyring_get)

    warnings: list[str] = []
    local = is_loopback(host)
    if not local and api_key is None:
        variable = api_key_env or DEFAULT_KEY_ENV
        raise ApiKeyError(
            f"Listening on {host} lets other machines reach this server, so it needs an API key.",
            fix=f"Set {variable}, or add --generate-api-key to create one and keep it in your "
            "keychain.",
            code=KEY_REQUIRED_CODE,
        )
    if api_key is not None and (weakness := key_weakness(api_key.value)):
        if not local:
            raise ApiKeyError(
                f"The API key is too weak: {weakness}.",
                fix="Use --generate-api-key for a strong one.",
            )
        warnings.append(f"The API key is weak ({weakness}); --generate-api-key makes a strong one.")
    if not local and tls_cert is None:
        warnings.append(
            "Traffic (prompts, answers and the API key) is not encrypted on the network. Add "
            "--tls-cert and --tls-key, or put Brasskey behind a TLS reverse proxy."
        )
    if host in WILDCARD_HOSTS:
        warnings.append(f"Listening on all network interfaces ({host}).")
    return ServeSecurity(
        host=host,
        api_key=api_key,
        tls_cert=tls_cert,
        tls_key=tls_key,
        generated=generate,
        stored_in_keyring=stored,
        warnings=tuple(warnings),
        key_env=api_key_env or DEFAULT_KEY_ENV,
    )
