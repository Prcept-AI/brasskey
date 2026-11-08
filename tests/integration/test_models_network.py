# SPDX-License-Identifier: Apache-2.0
"""Opt-in checks against the real Hugging Face Hub: ``uv run pytest -m network``.

They catch catalog entries whose pinned revision or files disappeared upstream (then rerun
scripts/refresh_catalog.py) and verify the real bandwidth probe.
"""

import pytest

from brasskey.models.catalog import load_catalog
from brasskey.models.download import measure_bandwidth
from brasskey.models.hub import RealHub

pytestmark = pytest.mark.network

ARTIFACTS = [artifact for model in load_catalog().models for artifact in model.artifacts]


@pytest.mark.parametrize("artifact", ARTIFACTS, ids=lambda a: f"{a.repo_id}@{a.short_revision}")
def test_pinned_catalog_files_still_exist(artifact):
    info = RealHub().repo_info(artifact.repo_id, revision=artifact.revision)
    assert info.revision == artifact.revision
    upstream = {file.path: file for file in info.files}
    for file in artifact.files:
        assert file.path in upstream
        assert upstream[file.path].size == file.size_bytes
        if file.sha256:
            assert upstream[file.path].sha256 == file.sha256


def test_real_bandwidth_probe():
    artifact = ARTIFACTS[0]
    largest = max(artifact.files, key=lambda file: file.size_bytes)
    url = RealHub().file_url(artifact.repo_id, artifact.revision, largest.path)
    rate = measure_bandwidth(url)
    assert rate is not None
    assert rate > 0
