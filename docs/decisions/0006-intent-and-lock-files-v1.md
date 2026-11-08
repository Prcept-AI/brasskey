# 0006: brasskey.yaml and brasskey.lock, version 1

Date: 2026-11-08
Status: accepted

## Context

Project rule (CONTRIBUTING.md): two files make deployments reproducible: `brasskey.yaml` (intent)
and `brasskey.lock` (resolved plan and measured results), defined with Pydantic v2 and exported as
JSON Schema. ADR 0002 decided when they are written.

## Options

- **Intent scope:** a full goal language (users, latency SLOs, budget, hardware pools), or only
  fields `run` acts on today.
- **Lock portability:** record the exact command line (machine-specific paths), or record the inputs
  that shape the launch, with paths replaced by placeholders.
- **When to write the lock:** after planning, or after the engine has actually started.

## Decision

- **`brasskey.yaml` v1** (`config/intent.py`):
  - Fields: `version: 1`, `model` (catalog id or `hf:`; optional), `engine` (optional),
    `goal.context_length` (default 8192), `goal.concurrency` (default 1), and `serve.host`,
    `serve.port` and `serve.api_key_env`.
  - Unknown fields are errors, which catches typos. Errors are plain and name the field.
  - `serve.host` must be loopback until the security checks of milestone (f) exist.
    (Lifted in milestone f, ADR 0009: any host is valid intent, and the API-key rule applies at
    run time. `serve.tls` (`cert_file`, `key_file`) and the lock's `serve.api_key_source` and
    `serve.tls` were added, all optional and additive within `version: 1`.)
    `api_key_env` must look like a variable name, so a key pasted by mistake is rejected.
  - New goal fields will be added when Brasskey acts on them (additive, same `version: 1`).
- **`brasskey.lock` v1** (`config/lock.py`):
  - Generator and time, plus a SHA-256 of the intent it came from.
  - Hardware: a fingerprint and a short summary.
  - Model metadata and architecture, so a replay needs no network.
  - The full artifact: repo, commit, files with sizes and SHA-256.
  - Engine id and version, plus the launch arguments with the model file shown as `{model}`.
  - Serve settings, a compact memory estimate (labeled an estimate), and the measurement:
    `started` and `startup_seconds`.
  - It never contains absolute paths, hostnames or secrets.
- **Writing and replaying:**
  - `brasskey.yaml` is written only by `--save`, never over an existing one.
  - `brasskey.lock` is written after the engine reports ready, with the measured start time,
    when `--save` is given or a `brasskey.yaml` is present. `--dry-run --save` writes a lock with
    `started: false`.
  - `brasskey run` without a model replays `brasskey.lock` when present, unless
    `brasskey.yaml` changed (its hash differs) or `--engine` / `--context` override it.
  - A replay re-estimates memory on the current machine. If the hardware fingerprint differs it
    says so, and it refuses plainly if the model no longer fits.
- **JSON Schemas:** `scripts/export_schemas.py` writes the schemas to `schemas/`. A test fails if
  they drift from the models.

## Consequences

- A committed lock reproduces the same files and engine settings on another machine. Memory is
  re-checked there, never assumed.
- Re-running from a lock doesn't rewrite it, so there's no churn in version control. Re-planning
  happens when the intent changes.
- Engine versions can differ on replay (the lock records the version it was written with).
  Strict version pinning can come with the registry, if design partners need it.
