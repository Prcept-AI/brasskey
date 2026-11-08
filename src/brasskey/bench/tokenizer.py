# SPDX-License-Identifier: Apache-2.0
"""Which tokenizer GuideLLM uses to build prompts of an exact token length.

In order:
1. ``--tokenizer``: a local folder, or a Hugging Face id (GuideLLM downloads it);
2. a catalog model: its tokenizer files, pinned and hashed in the catalog. Cached copies are
   used as they are (the MLX and safetensors versions include them). Otherwise just those files
   (a few MB, not the weights) are downloaded and verified first, after the user confirms;
3. otherwise a plain error asking for ``--tokenizer``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from brasskey.bench.guidellm import BenchError
from brasskey.models.fetch import fetch_artifact
from brasskey.models.hub import HubClient
from brasskey.models.types import Catalog, ModelArtifact

logger = logging.getLogger(__name__)

TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "config.json",
    }
)
REQUIRED = frozenset({"tokenizer_config.json"})


@dataclass(frozen=True)
class TokenizerChoice:
    source: str  # local folder or Hugging Face id
    local: bool
    description: str
    download: ModelArtifact | None = None  # tokenizer files to fetch first

    @property
    def download_bytes(self) -> int:
        return self.download.download_bytes if self.download is not None else 0


def _tokenizer_files(artifact: ModelArtifact) -> ModelArtifact | None:
    files = tuple(file for file in artifact.files if file.path in TOKENIZER_FILES)
    if not REQUIRED.issubset(file.path for file in files):
        return None
    return artifact.model_copy(update={"files": files})


def choose_tokenizer(
    model_name: str, explicit: str | None, *, catalog: Catalog, hub: HubClient
) -> TokenizerChoice:
    if explicit is not None:
        folder = Path(explicit).expanduser()
        if folder.is_dir():
            return TokenizerChoice(str(folder.resolve()), True, f"from {folder}")
        return TokenizerChoice(explicit, False, f"{explicit} from Hugging Face")

    model = next((entry for entry in catalog.models if entry.id == model_name.lower()), None)
    if model is None:
        # hf:org/name-GGUF:Q8_0 -> org/name (GGUF repos usually lack the tokenizer files).
        repo = model_name.removeprefix("hf:").split(":")[0] if model_name.startswith("hf:") else ""
        hint = repo[: -len("-GGUF")] if repo.upper().endswith("-GGUF") else repo
        example = hint or "Qwen/Qwen3-4B"
        raise BenchError(
            f"Brasskey doesn't know which tokenizer {model_name!r} uses.",
            fix=f"Pass --tokenizer with its Hugging Face id or a local folder, e.g. "
            f"--tokenizer {example}",
            code="BK-E-BENCH-TOKENIZER",
        )
    candidates = [subset for a in model.artifacts if (subset := _tokenizer_files(a)) is not None]
    for subset in candidates:
        paths = [hub.cached_path(subset.repo_id, subset.revision, f.path) for f in subset.files]
        if all(path is not None for path in paths):
            folder = paths[0].parent if paths[0] is not None else Path()
            logger.debug("Tokenizer for %s cached in %s", model.id, folder)
            return TokenizerChoice(str(folder), True, f"{subset.repo_id} (cached)")
    if not candidates:
        raise BenchError(
            f"The catalog has no tokenizer files for {model.display_name}.",
            fix=f"Pass --tokenizer {model.source_repo}",
            code="BK-E-BENCH-TOKENIZER",
        )
    smallest = min(candidates, key=lambda subset: subset.download_bytes)
    return TokenizerChoice(
        "", True, f"{smallest.repo_id} (tokenizer files only)", download=smallest
    )


def fetch_tokenizer(choice: TokenizerChoice, hub: HubClient) -> TokenizerChoice:
    """Download and verify the tokenizer files a choice still needs."""
    if choice.download is None:
        return choice
    local = fetch_artifact(choice.download, hub=hub)
    return TokenizerChoice(str(local.snapshot_dir), True, choice.description)
