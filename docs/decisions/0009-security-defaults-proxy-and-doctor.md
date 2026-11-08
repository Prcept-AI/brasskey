# 0009: Security defaults, the local proxy, and `doctor --security`

Date: 2026-11-08
Status: accepted

## Context

Milestone (f) turns the v0.1 security commitments (CONTRIBUTING.md, ADR 0002) into behavior:

- servers bind to `127.0.0.1` by default;
- any other address needs an API key;
- keys come only from the environment or the OS keychain;
- prompts are never logged;
- `brasskey doctor --security` explains the machine's exposure in plain words.

ADR 0008 accepted a thin local proxy so these rules hold the same way for every engine, MLX
included (`mlx_lm.server` has no API keys).

## Decision

### Serving: the proxy

Every `brasskey run` server is two processes:

```text
client ──> Brasskey proxy (--host:--port, the address users see)
             └──> engine on 127.0.0.1:<random free port> (private)
```

- The proxy is uvicorn + Starlette (both BSD-3-Clause) on a thread inside the `brasskey run`
  process. The engine is still a separate process (the rule against importing engines holds).
- It enforces, in this order:
  1. a browser `Origin` must be one of the server's own addresses (otherwise 403
     `origin_not_allowed`);
  2. the API key, when one is configured, as `Authorization: Bearer` or `x-api-key`, compared in
     constant time (otherwise 401 `invalid_api_key`);
  3. request bodies at most 32 MiB (otherwise 413 `request_too_large`);
  4. only the served model: `/v1/models` lists it alone, a request naming another model gets 404
     `model_not_found`, and a request with no model name gets the served one.
- Responses stream through unchanged (server-sent events included).
- `/health` needs no key and reveals only the model id. `/metrics` needs the key.
- Client credentials (`Authorization`, `x-api-key`, cookies, `Origin`, `Referer`) never reach the
  engine.
- For engines with their own key support (llama.cpp), Brasskey also generates a random internal
  key per run and passes it through the engine's environment, so another local process can't
  skip the proxy by calling the private port. MLX has no key support; its private port is
  loopback-only, and only the served model is reachable through the proxy.
- uvicorn's access log is off and its loggers are silenced: request paths, headers and bodies are
  never logged.
- Concurrency is left to the engines, which queue beyond their slot count (llama.cpp
  `--parallel`, the MLX limits from ADR 0007). A proxy-side limit can come later if measurements
  show a need.

### Keys and TLS

- **Where keys come from:**
  - `--api-key-env NAME` reads only that variable.
  - Otherwise Brasskey reads `BRASSKEY_API_KEY`, then the OS keychain (service `brasskey`,
    account `api-key`, through `keyring`, MIT).
  - Keys never appear in argv, logs, state files or `brasskey.lock`. Every key is registered with
    a log filter that replaces it with `[redacted]`. The lock records only where the key came from
    (`api_key_source`, such as `env:BRASSKEY_API_KEY` or `keyring`).
- **`--generate-api-key`:**
  - creates `bk-` plus 32 random bytes (URL-safe base64) and shows it once, on stderr (in
    `--json` mode, in `data.security.generated_api_key`);
  - stores it in the keychain;
  - with `--api-key-env NAME`, or when no keychain is available, does not store it, and tells the
    user to export the variable before the next run;
  - says so when the new key replaces one saved earlier, because clients with the old key are
    refused from then on;
  - does nothing in a dry run: the plan says a new key is created when it starts, and its
    `Next:` keeps `--generate-api-key`. A dry run never changes the keychain.
- **Exposure rules:**

  | Address | Key | Result |
  | --- | --- | --- |
  | loopback | none | allowed; doctor notes it |
  | loopback | weak | allowed, with a warning |
  | any other | none | refused before any download (`BK-E-API-KEY-REQUIRED`; `Next:` repeats the command with `--generate-api-key`) |
  | any other | weak (under 24 characters, or fewer than 8 distinct ones) | refused |
  | any other | strong | allowed, with a warning if it isn't encrypted and another if it listens on all interfaces |

- **TLS:** `--tls-cert` and `--tls-key` (or `serve.tls` in `brasskey.yaml`) make the proxy serve
  HTTPS. Both files are required. TLS is recommended off-loopback but not required: many teams
  terminate TLS at a reverse proxy.
- `brasskey.yaml` may now name any `serve.host`. The key rule applies at run time, because keys
  are never written to files.

### `brasskey doctor --security [URL]`

Read-only checks. It changes nothing and contacts only this machine's own ports, or the URL the
user names.

| ID | Check |
| --- | --- |
| BK-SEC-001 | Brasskey servers: address, key and TLS of each running server |
| BK-SEC-002 | Model servers need a key: listening model servers (by process name, image or known port) probed for `/v1/models` without a key |
| BK-SEC-003 | Web pages can't use model servers: probed with a foreign `Origin`; `*` or a reflected origin warns |
| BK-SEC-004 | Public network address: a public IP plus all-interface listeners fails |
| BK-SEC-005 | Docker port publishing: containers publishing on `0.0.0.0` |
| BK-SEC-006 | Firewall: the macOS application firewall (Linux needs root, so it is skipped with a hint) |
| BK-SEC-007 | API keys on command lines: `--api-key`, or Docker's `-e NAME=value` for credential-like names, in any process's argv (the pid is shown, never the value) |
| BK-SEC-008 | Brasskey API key: present and strong |
| BK-SEC-009 | Telemetry: Brasskey sends none and turns it off for engines |
| BK-SEC-010 | Encryption, URL mode only: plain HTTP to another machine fails |

- A port number alone is never reported. A finding needs a known process name or Docker image,
  or a probe answer that looks like a models list. (macOS AirPlay listens on port 5000.)
- Brasskey's own private engine ports are skipped.
- Exit codes:
  - 3 when any check fails (`CHECK_FAILED`, ADR 0001);
  - 0 otherwise, warnings included.

  `--json` keeps `ok: true` (the command worked); the verdict is in `data.passed`,
  `data.summary` and the exit code.
- Unit tests never scan the real machine: conftest replaces the scan with a quiet fake.

## Consequences

- Every engine now meets the same rules, and MLX can serve beyond loopback with a key.
- The proxy adds one local hop (sub-millisecond on loopback) and two dependencies, uvicorn and
  Starlette, plus keyring. On Linux, keyring pulls in SecretStorage, jeepney and cryptography.
  All are MIT, BSD or Apache-2.0 compatible.
- The proxy is the seam the Rust gateway replaces later. The contract to keep is the table
  above, plus the JSON error shape `{"error": {"message", "type", "code"}}`.
- Superseded parts of earlier ADRs:
  - ADR 0002 ("MLX refuses non-localhost binds");
  - ADR 0006 ("`serve.host` must be loopback");
  - ADR 0007 ("API keys refused for MLX").

  Each now carries a note pointing here.
- Revisit:
  - a proxy-side concurrency limit;
  - per-client keys and key rotation (Team features, not v0.1);
  - Linux firewall checks without root.
