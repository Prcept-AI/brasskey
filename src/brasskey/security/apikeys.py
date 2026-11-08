# SPDX-License-Identifier: Apache-2.0
"""API keys for Brasskey servers, from environment variables or the OS keyring.

See the secure-defaults rule in CONTRIBUTING.md.

Rules:
- A key is required whenever a server listens on anything other than loopback.
- Keys are read from ``BRASSKEY_API_KEY`` (or the variable named in ``serve.api_key_env``), or
  from the OS keyring (macOS Keychain, Secret Service on Linux, Windows Credential Locker).
- ``--generate-api-key`` creates a strong key, stores it in the keyring when one is available,
  and shows it once. It is never written to logs, lock files, state files or argv.
- Every key in use is registered with the log redactor (see ``brasskey.log``).
"""

import logging
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from brasskey.errors import BrasskeyError
from brasskey.log import register_secret

logger = logging.getLogger(__name__)

DEFAULT_KEY_ENV = "BRASSKEY_API_KEY"
KEYRING_SERVICE = "brasskey"
KEYRING_USERNAME = "api-key"
KEY_PREFIX = "bk-"
MIN_KEY_LENGTH = 24


class ApiKeyError(BrasskeyError):
    default_code = "BK-E-API-KEY"


@dataclass(frozen=True, slots=True)
class ApiKey:
    value: str = field(repr=False)  # never shown by repr() or in tracebacks
    source: str  # "env:NAME" or "keyring"

    def __post_init__(self) -> None:
        register_secret(self.value)


KeyringGet = Callable[[str, str], str | None]
KeyringSet = Callable[[str, str, str], None]


def default_keyring_get(service: str, username: str) -> str | None:
    try:
        import keyring  # MIT; loaded only when needed

        return keyring.get_password(service, username)
    except Exception:  # no backend (headless Linux) or a locked keychain: treat as "no key"
        logger.debug("Keyring unavailable for reading", exc_info=True)
        return None


def default_keyring_set(service: str, username: str, value: str) -> None:
    import keyring

    keyring.set_password(service, username, value)


def resolve_api_key(
    env_name: str | None,
    *,
    env: Mapping[str, str] | None = None,
    keyring_get: KeyringGet = default_keyring_get,
) -> ApiKey | None:
    """The API key to require, or None.

    With an explicit ``env_name`` only that variable counts. Otherwise ``BRASSKEY_API_KEY`` is
    checked first, then the OS keyring.
    """
    source = os.environ if env is None else env
    name = env_name or DEFAULT_KEY_ENV
    value = source.get(name)
    if value:
        return ApiKey(value=value, source=f"env:{name}")
    if env_name is None:
        stored = keyring_get(KEYRING_SERVICE, KEYRING_USERNAME)
        if stored:
            return ApiKey(value=stored, source="keyring")
    return None


def generate_api_key() -> str:
    """A new random key (~256 bits): ``bk-`` + 43 URL-safe characters."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def store_api_key(value: str, *, keyring_set: KeyringSet = default_keyring_set) -> bool:
    """Save the key in the OS keyring. False when no keyring is available."""
    try:
        keyring_set(KEYRING_SERVICE, KEYRING_USERNAME, value)
    except Exception:
        logger.debug("Keyring unavailable for writing", exc_info=True)
        return False
    return True


def key_weakness(value: str) -> str | None:
    """A plain description of why a key is too weak, or None."""
    if len(value) < MIN_KEY_LENGTH:
        return f"it is shorter than {MIN_KEY_LENGTH} characters"
    if len(set(value)) < 8:
        return "it repeats too few different characters"
    return None
