# SPDX-License-Identifier: Apache-2.0
"""A fake Hugging Face client implementing ``brasskey.models.hub.HubClient`` (no network)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brasskey.models.hub import ModelSourceError, RepoFile, RepoInfo
from brasskey.models.types import ModelArtifact
from fakes import FIXTURES

CONFIGS = FIXTURES / "configs"


def load_config(name: str) -> dict[str, Any]:
    return json.loads((CONFIGS / f"{name}.json").read_text())


def sha(n: int) -> str:
    return f"{n:064x}"


class FakeHub:
    def __init__(
        self,
        repos: dict[str, RepoInfo] | None = None,
        configs: dict[str, dict[str, Any]] | None = None,
        *,
        cached: dict[str, int] | None = None,
        cache_dir: Path = Path("fake-hf-cache"),  # only written when content is provided
        content: dict[str, bytes] | None = None,
        corrupt: bool = False,
    ) -> None:
        self.repos = repos or {}
        self.configs = configs or {}
        self.cached = cached or {}
        self._cache_dir = cache_dir
        self.urls: list[str] = []
        self.content = content or {}  # file path -> bytes served by download_file
        self.corrupt = corrupt  # serve different bytes than the checksum expects
        self.present: dict[tuple[str, str], Path] = {}
        self.downloads: list[str] = []

    def repo_info(self, repo_id: str, revision: str | None = None) -> RepoInfo:
        if repo_id not in self.repos:
            raise ModelSourceError(f"Hugging Face has no model repo named {repo_id}.")
        return self.repos[repo_id]

    def read_config(self, repo_id: str, revision: str) -> dict[str, Any]:
        return self.configs[repo_id]

    def cached_bytes(self, artifact: ModelArtifact) -> int:
        present = sum(
            path.stat().st_size
            for (repo, _name), path in self.present.items()
            if repo == artifact.repo_id
        )
        return present or self.cached.get(artifact.repo_id, 0)

    def cached_path(self, repo_id: str, revision: str, path: str) -> Path | None:
        return self.present.get((repo_id, path))

    def download_file(
        self, repo_id: str, revision: str, path: str, size: int, on_bytes: Callable[[int], None]
    ) -> Path:
        data = self.content[path]
        if self.corrupt:
            data = bytes(reversed(data))
        target = self._cache_dir / repo_id.replace("/", "--") / path
        target.parent.mkdir(parents=True, exist_ok=True)
        half = len(data) // 2
        target.write_bytes(data[:half])
        on_bytes(half)
        target.write_bytes(data)
        on_bytes(len(data))
        self.present[(repo_id, path)] = target
        self.downloads.append(path)
        return target

    def add_cached_blob(self, repo_id: str, path: str, data: bytes, sha256: str) -> Path:
        """Lay out a file like the real cache: snapshots/<path> -> blobs/<sha256>."""
        root = self._cache_dir / repo_id.replace("/", "--")
        blob = root / "blobs" / sha256
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = root / "snapshots" / path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        self.present[(repo_id, path)] = link
        return link

    def cache_dir(self) -> Path:
        return self._cache_dir

    def file_url(self, repo_id: str, revision: str, path: str) -> str:
        url = f"https://hf.example/{repo_id}/resolve/{revision}/{path}"
        self.urls.append(url)
        return url


def gguf_repo(repo_id: str, base: str | None, quants: dict[str, int]) -> RepoInfo:
    stem = repo_id.split("/")[-1].removesuffix("-GGUF")
    files = tuple(
        RepoFile(f"{stem}-{quant}.gguf", size, sha(index + 1))
        for index, (quant, size) in enumerate(quants.items())
    )
    return RepoInfo(
        repo_id=repo_id,
        revision="a" * 40,
        files=(RepoFile("README.md", 2000), *files),
        license="apache-2.0",
        base_model=base,
    )


def safetensors_repo(
    repo_id: str, weights: int, *, params: int | None, tags: tuple[str, ...] = ()
) -> RepoInfo:
    return RepoInfo(
        repo_id=repo_id,
        revision="b" * 40,
        files=(
            RepoFile("config.json", 700),
            RepoFile("model.safetensors", weights, sha(99)),
            RepoFile("tokenizer.json", 11_000_000, sha(98)),
            RepoFile("README.md", 3000),
            RepoFile("original/consolidated.pth", weights, sha(97)),
        ),
        license="apache-2.0",
        params_total=params,
        tags=tags,
    )
