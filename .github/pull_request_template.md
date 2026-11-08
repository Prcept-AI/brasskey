## What and why

<!-- What does this change do, and which issue or milestone does it serve? -->

## How it was tested

<!-- Commands you ran and what you saw. Note any opt-in tests (-m gpu / engine / network). -->

## Checklist

- [ ] In scope for the current milestone (see CONTRIBUTING.md and the README's Status section)
- [ ] Tests added or updated; `uv run pytest` passes without a GPU
- [ ] `uv run ruff check . && uv run ruff format --check .` and `uv run pyright` pass
- [ ] New `.py` files start with `# SPDX-License-Identifier: Apache-2.0`
- [ ] No new dependency, or it was approved and its license (Apache-2.0 / MIT / BSD compatible) is
      recorded in NOTICE
- [ ] Commands keep the CLI contract: `--json`, plain errors with a fix, a `Next:` line
- [ ] No prompt content, API keys or tokens are logged or stored
- [ ] Notable decisions recorded as an ADR in `docs/decisions/`
- [ ] User-visible changes noted under "Unreleased" in `CHANGELOG.md`
