# SPDX-License-Identifier: Apache-2.0
"""API keys, the exposure policy, and log redaction."""

import logging
from pathlib import Path

import pytest

from brasskey.log import configure_logging, redact
from brasskey.security.apikeys import (
    KEYRING_SERVICE,
    KEYRING_USERNAME,
    ApiKey,
    ApiKeyError,
    generate_api_key,
    key_weakness,
    resolve_api_key,
    store_api_key,
)
from brasskey.security.policy import decide_security

STRONG = "bk-7Hc2Lq9XzP4mVw8RtY6nJ3aK5sD1fG0e"


def no_keyring(_service, _user):
    return None


def no_save(_service: str, _user: str, _value: str) -> None:
    return None


# --- keys ----------------------------------------------------------------------------------------


def test_default_variable_first_then_the_keyring():
    assert resolve_api_key(
        None, env={"BRASSKEY_API_KEY": STRONG}, keyring_get=no_keyring
    ) == ApiKey(STRONG, "env:BRASSKEY_API_KEY")
    stored = resolve_api_key(None, env={}, keyring_get=lambda s, u: STRONG)
    assert stored is not None
    assert stored.source == "keyring"
    assert resolve_api_key(None, env={}, keyring_get=no_keyring) is None


def test_an_explicit_variable_never_falls_back_to_the_keyring():
    assert resolve_api_key("TEAM_KEY", env={}, keyring_get=lambda s, u: STRONG) is None
    found = resolve_api_key("TEAM_KEY", env={"TEAM_KEY": STRONG}, keyring_get=no_keyring)
    assert found is not None
    assert found.source == "env:TEAM_KEY"


def test_generated_keys_are_long_random_and_prefixed():
    first, second = generate_api_key(), generate_api_key()
    assert first.startswith("bk-")
    assert len(first) >= 40
    assert first != second
    assert key_weakness(first) is None


def test_store_uses_the_brasskey_entry_and_reports_failure():
    saved: list[tuple[str, str, str]] = []
    assert store_api_key(STRONG, keyring_set=lambda s, u, v: saved.append((s, u, v))) is True
    assert saved == [(KEYRING_SERVICE, KEYRING_USERNAME, STRONG)]

    def broken(*_args):
        raise RuntimeError("no backend")

    assert store_api_key(STRONG, keyring_set=broken) is False


@pytest.mark.parametrize(("key", "weak"), [("short", True), ("a" * 30, True), (STRONG, False)])
def test_key_weakness(key, weak):
    assert (key_weakness(key) is not None) is weak


def test_keys_are_hidden_from_repr_and_redacted_from_logs(capsys):
    key = ApiKey(value="bk-super-secret-value-123456", source="env:X")
    assert "super-secret" not in repr(key)
    configure_logging(debug=True, json_mode=True)
    logging.getLogger("brasskey.test").debug("using key %s now", key.value)
    assert "bk-super-secret-value-123456" not in capsys.readouterr().err
    assert redact("x bk-super-secret-value-123456 y") == "x [redacted] y"
    configure_logging(debug=False, json_mode=False)


# --- policy --------------------------------------------------------------------------------------


def decide(host, *, env=None, generate=False, tls_cert=None, tls_key=None, keyring=no_keyring,
           saved=None):  # fmt: skip
    return decide_security(
        host, api_key_env=None, generate=generate, tls_cert=tls_cert, tls_key=tls_key,
        env=env or {}, keyring_get=keyring,
        keyring_set=(lambda s, u, v: saved.append(v)) if saved is not None else no_save,
    )  # fmt: skip


def test_loopback_needs_no_key():
    security = decide("127.0.0.1")
    assert security.api_key is None
    assert security.access_label == "no API key (this machine only)"
    assert security.warnings == ()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::"])  # noqa: S104
def test_other_addresses_require_a_key(host):
    with pytest.raises(ApiKeyError) as caught:
        decide(host)
    assert "needs an API key" in caught.value.message
    assert "--generate-api-key" in (caught.value.fix or "")


def test_key_from_env_allows_the_network_but_warns_about_plain_http():
    security = decide("0.0.0.0", env={"BRASSKEY_API_KEY": STRONG})  # noqa: S104
    assert security.access_label == "API key required (from BRASSKEY_API_KEY)"
    assert any("not encrypted" in warning for warning in security.warnings)
    assert any("all network interfaces" in warning for warning in security.warnings)
    assert security.client_host == "127.0.0.1"


def test_weak_keys_are_refused_on_the_network_and_flagged_locally():
    with pytest.raises(ApiKeyError, match="too weak"):
        decide("192.168.1.20", env={"BRASSKEY_API_KEY": "short-key"})
    local = decide("127.0.0.1", env={"BRASSKEY_API_KEY": "short-key-1"})
    assert any("weak" in warning for warning in local.warnings)


def test_generate_saves_to_the_keyring():
    saved: list[str] = []
    security = decide("0.0.0.0", generate=True, saved=saved)  # noqa: S104
    assert security.generated
    assert security.stored_in_keyring
    assert security.api_key is not None
    assert saved == [security.api_key.value]
    assert security.access_label == "API key required (from your keychain)"


def test_tls_needs_both_existing_files(tmp_path: Path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text("cert")
    key.write_text("key")
    with pytest.raises(ApiKeyError, match="both"):
        decide("127.0.0.1", tls_cert=cert)
    with pytest.raises(ApiKeyError, match="not found"):
        decide("127.0.0.1", tls_cert=cert, tls_key=tmp_path / "missing.pem")
    security = decide("0.0.0.0", env={"BRASSKEY_API_KEY": STRONG}, tls_cert=cert, tls_key=key)  # noqa: S104
    assert security.scheme == "https"
    assert not any("not encrypted" in warning for warning in security.warnings)


def test_origins_are_only_the_servers_own_addresses():
    security = decide("192.168.1.20", env={"BRASSKEY_API_KEY": STRONG})
    origins = security.origins(8000)
    assert "http://192.168.1.20:8000" in origins
    assert "http://127.0.0.1:8000" in origins
    assert all(origin.endswith(":8000") for origin in origins)
