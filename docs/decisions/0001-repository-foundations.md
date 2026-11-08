# 0001: Repository foundations: layout, tooling and the CLI output contract

Date: 2026-11-08
Status: accepted

## Context

Milestone (a) of v0.1 sets up the repository every later milestone builds on. The project rules fix
the stack (Python 3.11+, uv, Typer + Rich, Pydantic v2, pytest, ruff, pyright) and the supply-chain
rules (CONTRIBUTING.md); a few choices inside those rules needed deciding.

## Options

- **Build backend:** hatchling (mature, PEP 639 license metadata) or uv_build (newer).
- **Dependency freshness:** take the newest releases, or wait a cooldown period before adopting
  new releases (the LiteLLM incident of Mar 2026 shipped through a freshly compromised tool).
- **Python interpreter:** let uv use whatever interpreter it discovers first (which picks up an
  active conda environment), or uv-managed builds only.
- **Error handling:** let Typer/Click print errors in standalone mode, or own the boundary.
- **Click:** Typer 0.27 vendors Click as `typer._click`; import that private module, or depend
  only on Typer's public API.

## Decision

- **Layout:** `src/brasskey/` with the planned packages (`cli`, `hardware`, `engines`, `models`,
  `bench`, `security`, `config`, `output`) plus two approved additions arriving in later
  milestones: `planner/` (pure engine and model choice) and `runtime/` (process lifecycle,
  state files). `scripts/` holds developer tools; `schemas/` will hold exported JSON Schemas.
- **Build and lock:** hatchling (exact-pinned) builds; `uv.lock` is committed; dev tools are pinned
  exactly; runtime dependencies use lower bounds equal to the tested versions.
- **Cooldown:** `[tool.uv] exclude-newer = "7 days"` and Dependabot `cooldown: 7 days`. This needs
  uv 0.13.6+ (`required-version`), which also pins CI.
- **Interpreter:** `python-preference = "only-managed"` for reproducible environments.
- **Lint and types:** ruff with bugbear, bandit (`S`), pyupgrade, pytest style and a ban on
  importing engines (`TID251`: vllm, sglang, mlx, mlx_lm, llama_cpp, torch, guidellm); pyright
  strict for `src/` and `scripts/`, standard for tests.
- **Tests:** opt-in markers `gpu`, `engine`, `network` are skipped unless named with `-m`.
- **CI:** jobs for lint/types/licenses, a test matrix (ubuntu-24.04 and macos-15 × Python
  3.11-3.14) and an audit (pip-audit on hashed exported requirements, zizmor on workflows). No
  default token permissions; actions pinned by SHA; checkout without persisted credentials.
  GitHub's dependency-review action was left out because it fails on private repositories
  without Advanced Security; pip-audit and Dependabot alerts cover known vulnerabilities.
- **License policy:** `scripts/check_licenses.py` fails CI on GPL/AGPL/SSPL/BUSL or unknown
  licenses (SPDX expressions parsed with AND/OR/WITH); MPL-2.0 is allowed and reported.
- **CLI boundary:** `brasskey.cli.app.main` runs Click in non-standalone mode and renders every
  outcome itself, using only Typer's public API (`TyperException`, `Exit`, `Abort`).
- **Output contract:**
  - Commands return a `CommandResult` (JSON data, Rich view, `NextStep`, exit code).
  - `--json` prints exactly one envelope `{"schema": "brasskey.output/v1", "command", "ok",
    "data", "error", "next"}` to stdout; logs go to stderr.
  - `ok` means the command completed; verdicts live in `data` and the exit code.
  - Flags work before or after the command.
  - Text is built with `rich.text.Text`, not markup, so brackets print literally.
- **Errors:** `BrasskeyError(message, fix, code="BK-E-...", exit_code, next_step)`; unexpected
  exceptions print a short message and a `brasskey --debug ...` rerun hint; tracebacks (without
  local variables, which could hold secrets) only with `--debug`.
- **Exit codes:** 0 OK/PASS, 1 error, 2 usage, 3 check FAILED, 4 INCONCLUSIVE, 130 Ctrl-C,
  141 output pipe closed by the reader (e.g. `| head`), without a traceback.

## Consequences

- Contributors need uv 0.13.6+; older uv cannot read the relative `exclude-newer`.
- New releases (including security fixes) reach us after 7 days. For an urgent fix, override per
  package with `exclude-newer-package` in `pyproject.toml` and note why in the pull request.
- Every new command must be added to `SAMPLE_ARGS` in `tests/unit/test_cli_contract.py`, which
  enforces `--json` and the `Next:` footer automatically.
- Revisit when the release pipeline lands (trusted publishing, Sigstore, SBOM, checksums).
