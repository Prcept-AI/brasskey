# Releasing Brasskey

Releases are built and published by `.github/workflows/release.yml` (design:
[ADR 0012](docs/decisions/0012-release-pipeline.md)). No one uploads from a laptop, and no PyPI
token exists anywhere: PyPI trusts the workflow itself (trusted publishing).

## One-time setup

1. **PyPI and TestPyPI trusted publishers.** On <https://pypi.org/manage/account/publishing/> add a
   *pending publisher*, and the same on <https://test.pypi.org/manage/account/publishing/>:

   | Field | PyPI | TestPyPI |
   | --- | --- | --- |
   | PyPI project name | `brasskey` | `brasskey` |
   | Owner | `Prcept-AI` | `Prcept-AI` |
   | Repository name | `brasskey` | `brasskey` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

   A pending publisher doesn't reserve the name; the first upload does.
2. **GitHub environments** (Settings > Environments): create `pypi` and `testpypi`.
   - For `pypi`, limit deployments to tags matching `v*`.
   - Add a required reviewer (on private repositories this needs a paid GitHub plan; it is free
     once the repository is public).
3. **Public repository.** GitHub build provenance (artifact attestations) needs a public
   repository; the workflow skips it while the repository is private.

## Rehearse on TestPyPI

Actions > Release > Run workflow (any branch). It builds, checks, and publishes to TestPyPI,
which accepts each version only once, so rehearse with a `.devN` version if needed. Then:

```bash
uv venv /tmp/try && uv pip install --python /tmp/try/bin/python \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ brasskey
/tmp/try/bin/brasskey --version
```

## Release

1. Set the version in `src/brasskey/__init__.py` (PEP 440: `0.1.0a1`, `0.1.0b1`, `0.1.0rc1`,
   `0.1.0`), update the README's install instructions if they changed, run
   `sh scripts/check_secrets.sh` (it must report no credentials), and merge to `main` with CI
   green.
2. Tag the merge commit and push the tag:

   ```bash
   git tag -a v0.1.0a1 -m "Brasskey 0.1.0a1"
   git push origin v0.1.0a1
   ```
3. Approve the `pypi` deployment when GitHub asks. The workflow then:
   - builds the wheel and sdist twice from that commit and compares them (reproducible builds);
   - checks them with `scripts/check_dist.py` (the tag must match the version);
   - installs the wheel into a clean environment and runs it;
   - publishes to PyPI with Sigstore-signed attestations;
   - records build provenance and creates the GitHub release with the wheel, sdist,
     `SHA256SUMS` and an SPDX SBOM. Alpha, beta, rc and dev versions are marked pre-release;
   - pushes the container image `ghcr.io/prcept-ai/brasskey:<version>` (linux/amd64 and
     linux/arm64, BuildKit provenance; `latest` only for final versions). The first time, set
     the package's visibility to public in the organization's package settings.

4. Update the Homebrew tap ([ADR 0015](docs/decisions/0015-homebrew-tap.md)) once the version is
   on PyPI:

   ```bash
   git clone https://github.com/Prcept-AI/homebrew-tap.git && cd homebrew-tap
   git switch -c brasskey-<version>
   uv run --project ../brasskey python ../brasskey/scripts/homebrew_formula.py <version> \
     > Formula/brasskey.rb
   git commit -am "brasskey <version>" && git push -u origin brasskey-<version>
   gh pr create --fill
   ```

   When the pull request's test-bot checks are green (bottles for macOS and Linux), run the tap's
   "brew pr-pull" workflow with the pull request number: it publishes the bottles and merges.
5. Start the next version on `main`: set `__version__` to the next development version (for
   example `0.1.0a3.dev0` after `0.1.0a2`), so builds from `main` can't be mistaken for the
   release.

A failed run publishes nothing that a rerun can't finish; PyPI rejects re-uploading an existing
version, so fix forward with the next version number.

## Check a build locally

```bash
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build \
  --build-constraints scripts/build-constraints.txt --require-hashes --no-create-gitignore
python3 scripts/check_dist.py dist
```

Users can verify what they download as described in [SECURITY.md](SECURITY.md#verifying-a-release).
