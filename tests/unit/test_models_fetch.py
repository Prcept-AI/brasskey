# SPDX-License-Identifier: Apache-2.0
"""Downloading into the cache and verifying checksums before any engine sees the files."""

import pytest

from brasskey.models import fetch
from brasskey.models.fetch import IntegrityError, fetch_artifact
from fakes.engine import TINY_ARTIFACT, TINY_CONTENT, sha256_of
from fakes.hub import FakeHub

PATH = TINY_ARTIFACT.files[0].path
DATA = TINY_CONTENT[PATH]


def test_downloads_verifies_and_reports_progress(tmp_path):
    hub = FakeHub(cache_dir=tmp_path, content=TINY_CONTENT)
    progress: list[tuple[int, int]] = []
    verified: list[tuple[int, int]] = []
    local = fetch_artifact(
        TINY_ARTIFACT,
        hub=hub,
        on_progress=lambda done, total: progress.append((done, total)),
        on_verify=lambda done, total: verified.append((done, total)),
    )
    assert hub.downloads == [PATH]
    assert local.primary_file.read_bytes() == DATA
    assert progress[-1] == (len(DATA), len(DATA))
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert verified[-1] == (len(DATA), len(DATA))  # fresh files are always hashed


def test_corrupt_download_is_refused(tmp_path):
    hub = FakeHub(cache_dir=tmp_path, content=TINY_CONTENT, corrupt=True)
    with pytest.raises(IntegrityError) as caught:
        fetch_artifact(TINY_ARTIFACT, hub=hub)
    assert "won't run it" in caught.value.message


def test_cached_file_is_checked_by_blob_name_without_rehashing(tmp_path, monkeypatch):
    hub = FakeHub(cache_dir=tmp_path)
    hub.add_cached_blob(TINY_ARTIFACT.repo_id, PATH, DATA, sha256_of(DATA))

    def no_hashing(*_args, **_kwargs):
        pytest.fail("cached files with a matching blob name must not be re-hashed")

    monkeypatch.setattr(fetch, "sha256_file", no_hashing)
    local = fetch_artifact(TINY_ARTIFACT, hub=hub)
    assert hub.downloads == []
    assert local.files[0].name == PATH


def test_cached_file_with_the_wrong_size_is_refused(tmp_path):
    hub = FakeHub(cache_dir=tmp_path)
    hub.add_cached_blob(TINY_ARTIFACT.repo_id, PATH, DATA[:10], sha256_of(DATA))
    with pytest.raises(IntegrityError) as caught:
        fetch_artifact(TINY_ARTIFACT, hub=hub)
    assert "the catalog expects" in caught.value.message


def test_cached_file_under_an_unexpected_blob_is_rehashed(tmp_path):
    hub = FakeHub(cache_dir=tmp_path)
    hub.add_cached_blob(TINY_ARTIFACT.repo_id, PATH, DATA, "f" * 64)  # name doesn't prove it
    fetch_artifact(TINY_ARTIFACT, hub=hub)  # content still matches, so it passes
