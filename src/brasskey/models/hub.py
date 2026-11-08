# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Hub access: repo metadata, configs, the local cache, and file URLs.

This is the only module that imports ``huggingface_hub`` (lazily, to keep the CLI fast). It sits
behind the :class:`HubClient` protocol so tests use a fake and never touch the network.

The same helpers build catalog artifacts at runtime (``hf:org/name``) and in
``scripts/refresh_catalog.py``, so catalog numbers and runtime numbers come from one code path.

Privacy (CONTRIBUTING.md): Hugging Face's own telemetry is turned off before the library loads, and
its warnings and progress bars are silenced so Brasskey controls what the user sees.
"""

import json
import logging
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from brasskey.engines.types import ModelFormat
from brasskey.errors import BrasskeyError
from brasskey.models.architecture import ArchitectureError, architecture_from_config
from brasskey.models.types import ArtifactFile, CatalogModel, ModelArtifact

logger = logging.getLogger(__name__)

# Must be set before huggingface_hub is imported anywhere in this process.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

GGUF_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0")
# Files needed to serve a safetensors or MLX repo (weights, configs, tokenizer, chat template).
KEEP_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".tiktoken", ".jinja")
_SPLIT_GGUF = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class ModelSourceError(BrasskeyError):
    """A model couldn't be found or read on Hugging Face."""

    default_code = "BK-E-MODEL-SOURCE"


@dataclass(frozen=True, slots=True)
class RepoFile:
    path: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RepoInfo:
    repo_id: str
    revision: str  # commit SHA
    files: tuple[RepoFile, ...]
    license: str | None = None
    gated: bool = False
    base_model: str | None = None
    params_total: int | None = None
    tags: tuple[str, ...] = ()


class HubClient(Protocol):
    def repo_info(self, repo_id: str, revision: str | None = None) -> RepoInfo: ...
    def read_config(self, repo_id: str, revision: str) -> dict[str, Any]: ...
    def cached_bytes(self, artifact: ModelArtifact) -> int: ...
    def cached_path(self, repo_id: str, revision: str, path: str) -> Path | None: ...
    def download_file(
        self, repo_id: str, revision: str, path: str, size: int, on_bytes: Callable[[int], None]
    ) -> Path: ...
    def cache_dir(self) -> Path: ...
    def file_url(self, repo_id: str, revision: str, path: str) -> str: ...


# --- shared artifact helpers --------------------------------------------------------------------


def _gguf_matches(path: str, quantization: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    tag = quantization.lower()
    return f"-{tag}." in name or f"-{tag}-" in name or f"_{tag}." in name


def select_files(
    files: Sequence[RepoFile], fmt: ModelFormat, gguf_quantization: str | None = None
) -> tuple[tuple[RepoFile, ...], str | None]:
    """Files to download for one artifact, and the GGUF quantization picked (if GGUF)."""
    if fmt == ModelFormat.GGUF:
        ggufs = [file for file in files if file.path.lower().endswith(".gguf")]
        wanted = [gguf_quantization] if gguf_quantization else list(GGUF_PREFERENCE)
        for quantization in wanted:
            chosen = tuple(file for file in ggufs if _gguf_matches(file.path, quantization))
            if chosen:
                return chosen, quantization.upper()
        if gguf_quantization:
            available = ", ".join(sorted(file.path for file in ggufs)) or "none"
            raise ModelSourceError(
                f"No {gguf_quantization} GGUF file in this repo (files: {available}).",
                fix="Pick one of the listed quantizations, e.g. hf:org/name:Q4_K_M.",
            )
        if not ggufs:
            raise ModelSourceError("This repo has no GGUF files.")
        by_path = sorted(ggufs, key=lambda file: file.path)
        single = [file for file in by_path if not _SPLIT_GGUF.search(file.path)]
        first = single[0] if single else by_path[0]
        return (first,), None
    kept = tuple(
        file for file in files if "/" not in file.path and file.path.lower().endswith(KEEP_SUFFIXES)
    )
    if not any(file.path.endswith(".safetensors") for file in kept):
        raise ModelSourceError("This repo has no safetensors weights at its top level.")
    return kept, None


def detect_quantization(
    fmt: ModelFormat, config: dict[str, Any] | None, gguf_quantization: str | None
) -> str:
    if fmt == ModelFormat.GGUF:
        return gguf_quantization or "gguf"
    config = config or {}
    raw: object = config.get("quantization_config") or config.get("quantization")
    quant = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    bits: object = quant.get("bits")
    if fmt == ModelFormat.MLX:
        return f"{bits}bit" if bits else "mlx"
    if method_value := quant.get("quant_method"):
        method = str(method_value).lower()
        return f"{method}-int{bits}" if method in {"awq", "gptq"} and bits else method
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "").lower()
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(dtype, dtype or "unknown")


def build_artifact(
    info: RepoInfo,
    fmt: ModelFormat,
    *,
    config: dict[str, Any] | None,
    gguf_quantization: str | None = None,
) -> ModelArtifact:
    files, picked = select_files(info.files, fmt, gguf_quantization)
    return ModelArtifact(
        format=fmt,
        repo_id=info.repo_id,
        revision=info.revision,
        quantization=detect_quantization(fmt, config, picked or gguf_quantization),
        files=tuple(ArtifactFile(path=f.path, size_bytes=f.size, sha256=f.sha256) for f in files),
    )


def detect_format(info: RepoInfo) -> ModelFormat:
    paths = [file.path.lower() for file in info.files]
    if any(path.endswith(".gguf") for path in paths):
        return ModelFormat.GGUF
    if "mlx" in info.tags:
        return ModelFormat.MLX
    if any(path.endswith(".safetensors") for path in paths):
        return ModelFormat.SAFETENSORS
    raise ModelSourceError(
        f"{info.repo_id} has no GGUF or safetensors weights Brasskey can run.",
        fix="Use a repo with GGUF (llama.cpp), MLX or safetensors (vLLM/SGLang) weights.",
    )


def parse_hf_spec(spec: str) -> tuple[str, str | None]:
    """``hf:org/name`` or ``hf:org/name:Q4_K_M`` -> (repo id, GGUF quantization or None)."""
    body = spec.removeprefix("hf:")
    repo_id, _, quantization = body.partition(":")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ModelSourceError(
            f"'{spec}' isn't a Hugging Face model name.",
            fix="Write it as hf:org/name, for example hf:Qwen/Qwen3-4B-GGUF:Q4_K_M.",
        )
    return repo_id, quantization or None


def resolve_hf_model(spec: str, hub: HubClient) -> CatalogModel:
    """Build an (unverified) catalog entry for any Hugging Face repo."""
    repo_id, quantization = parse_hf_spec(spec)
    info = hub.repo_info(repo_id)
    if info.gated:
        raise ModelSourceError(
            f"{repo_id} is gated: Hugging Face requires accepting its license first.",
            fix=f"Accept it at https://huggingface.co/{repo_id}, then set HF_TOKEN.",
        )
    fmt = detect_format(info)
    notes: list[str] = []
    if fmt == ModelFormat.GGUF:
        if not info.base_model:
            raise ModelSourceError(
                f"{repo_id} doesn't name its base model, so Brasskey can't read the architecture "
                "it needs for the memory estimate.",
                fix="Use a GGUF repo whose model card lists base_model, or the original repo.",
            )
        source = hub.repo_info(info.base_model)
        config = hub.read_config(source.repo_id, source.revision)
        params = source.params_total
    else:
        source = info
        config = hub.read_config(repo_id, info.revision)
        params = info.params_total
    artifact = build_artifact(info, fmt, config=config, gguf_quantization=quantization)
    if params is None:
        params = max(artifact.weights_bytes // 2, 1)
        notes.append("Parameter count estimated from file size (assuming 16-bit weights).")
    try:
        parsed = architecture_from_config(config, params_total=params)
    except ArchitectureError as error:
        raise ModelSourceError(
            f"Couldn't read the architecture of {source.repo_id}: {error}.",
            fix="Use a catalog model or a repo with a standard config.json.",
        ) from error
    notes.extend(parsed.notes)
    model_type = config.get("model_type")
    return CatalogModel(
        id=f"hf:{repo_id}",
        display_name=repo_id,
        family=model_type if isinstance(model_type, str) else "unknown",
        license=info.license or "unknown",
        source_repo=source.repo_id,
        verified=False,
        architecture=parsed.architecture,
        artifacts=(artifact,),
        notes=tuple(notes),
    )


# --- the real client ----------------------------------------------------------------------------


def _first_str(value: object) -> str | None:
    """A model-card field that may be a string or a list of strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        return items[0] if items and isinstance(items[0], str) else None
    return None


class _FileProgress:
    """Feeds one file's download progress to Brasskey's display.

    huggingface_hub accepts any class as ``tqdm_class``; it creates one or two bars per file (Xet
    downloads report "reconstructing" and "downloading bytes"). Only bars whose total is the file
    size count, so progress is never double-counted.
    """

    def __init__(self, size: int, on_bytes: Callable[[int], None]) -> None:
        self.size = size
        self.on_bytes = on_bytes
        self.bars: list[Any] = []
        self.lock = threading.Lock()

    def emit(self) -> None:
        with self.lock:
            sized = [bar.n for bar in self.bars if bar.total == self.size]
            done = max(sized or [bar.n for bar in self.bars] or [0])
        self.on_bytes(min(done, self.size))

    def bar_class(self) -> type:
        tracker = self

        class _Bar:
            def __init__(self, *_args: object, **kwargs: object) -> None:
                initial = kwargs.get("initial", 0)
                self.total = kwargs.get("total")
                self.n = initial if isinstance(initial, int) else 0
                with tracker.lock:
                    tracker.bars.append(self)
                tracker.emit()

            def update(self, n: object = 1) -> None:
                self.n += int(n) if isinstance(n, (int, float)) else 0
                tracker.emit()

            def reset(self, total: object = None) -> None:
                self.n = 0
                self.total = total

            def close(self) -> None:
                return None

            def refresh(self, *_args: object, **_kwargs: object) -> None:
                return None

            def set_description(self, *_args: object, **_kwargs: object) -> None:
                return None

            def set_postfix(self, *_args: object, **_kwargs: object) -> None:
                return None

            def __enter__(self) -> "_Bar":
                return self

            def __exit__(self, *_exc: object) -> None:
                self.close()

        return _Bar


def _quiet_huggingface_hub() -> None:
    """Silence huggingface_hub's own warnings and progress bars (Brasskey renders output)."""
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    disable_progress_bars()


class RealHub:
    """:class:`HubClient` backed by ``huggingface_hub`` (Apache-2.0)."""

    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        _quiet_huggingface_hub()

    def _error(self, error: Exception, repo_id: str) -> BrasskeyError:
        from huggingface_hub import errors as hf_errors

        logger.debug("Hugging Face error for %s: %s", repo_id, type(error).__name__)
        if isinstance(error, hf_errors.GatedRepoError):
            return ModelSourceError(
                f"{repo_id} is gated: Hugging Face requires accepting its license first.",
                fix=f"Accept it at https://huggingface.co/{repo_id}, then set HF_TOKEN.",
            )
        if isinstance(error, hf_errors.RepositoryNotFoundError):
            return ModelSourceError(
                f"Hugging Face has no model repo named {repo_id} (or it is private).",
                fix="Check the name on https://huggingface.co/models; for a private repo set "
                "HF_TOKEN.",
            )
        if isinstance(error, hf_errors.RevisionNotFoundError):
            return ModelSourceError(f"{repo_id} has no such revision.")
        if isinstance(error, (hf_errors.OfflineModeIsEnabled, hf_errors.LocalEntryNotFoundError)):
            return ModelSourceError(
                f"{repo_id} isn't in the local Hugging Face cache and Brasskey is offline.",
                fix="Connect to the internet and drop --offline (or unset HF_HUB_OFFLINE).",
            )
        return ModelSourceError(
            f"Couldn't reach Hugging Face for {repo_id} ({type(error).__name__}).",
            fix="Check your internet connection (and HF_ENDPOINT if you use a mirror), then retry.",
        )

    def repo_info(self, repo_id: str, revision: str | None = None) -> RepoInfo:
        if self.offline:
            raise ModelSourceError(
                f"Looking up {repo_id} needs the internet, but Brasskey is offline.",
                fix="Drop --offline, or pick a model from the Brasskey catalog.",
            )
        from huggingface_hub import HfApi

        try:
            info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        except Exception as error:  # translated into a plain-language error below
            raise self._error(error, repo_id) from error
        card: Any = info.card_data
        base = _first_str(card.get("base_model") if card is not None else None)
        license_name = _first_str(card.get("license") if card is not None else None)
        files = tuple(
            RepoFile(
                path=sibling.rfilename,
                size=int(sibling.size or (sibling.lfs.size if sibling.lfs else 0)),
                sha256=sibling.lfs.sha256 if sibling.lfs else None,
            )
            for sibling in info.siblings or []
        )
        return RepoInfo(
            repo_id=info.id,
            revision=str(info.sha),
            files=files,
            license=license_name,
            gated=bool(info.gated),
            base_model=base,
            params_total=info.safetensors.total if info.safetensors else None,
            tags=tuple(info.tags or ()),
        )

    def read_config(self, repo_id: str, revision: str) -> dict[str, Any]:
        # The signature mentions tqdm generics pyright can't fully resolve; the call is typed.
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

        try:
            path = hf_hub_download(
                repo_id, "config.json", revision=revision, local_files_only=self.offline
            )
        except Exception as error:
            raise self._error(error, repo_id) from error
        with Path(path).open(encoding="utf-8") as handle:
            loaded: Any = json.load(handle)
        if not isinstance(loaded, dict):
            raise ModelSourceError(f"{repo_id}/config.json is not a JSON object.")
        return dict(loaded)  # pyright: ignore[reportUnknownArgumentType]

    def cached_path(self, repo_id: str, revision: str, path: str) -> Path | None:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo_id, path, revision=revision)
        if isinstance(found, str) and Path(found).is_file():
            return Path(found)
        return None

    def cached_bytes(self, artifact: ModelArtifact) -> int:
        total = 0
        for file in artifact.files:
            found = self.cached_path(artifact.repo_id, artifact.revision, file.path)
            if found is not None:
                total += found.stat().st_size
        return total

    def download_file(
        self,
        repo_id: str,
        revision: str,
        path: str,
        size: int,
        on_bytes: Callable[[int], None],
    ) -> Path:
        if self.offline:
            raise ModelSourceError(
                f"{repo_id}/{path} isn't downloaded yet and Brasskey is offline.",
                fix="Drop --offline (or unset HF_HUB_OFFLINE) to download it.",
            )
        # The signature mentions tqdm generics pyright can't fully resolve; the call is typed.
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

        progress = _FileProgress(size, on_bytes)
        try:
            local = hf_hub_download(
                repo_id, path, revision=revision, tqdm_class=cast(Any, progress.bar_class())
            )
        except Exception as error:  # translated into a plain-language error
            raise self._error(error, repo_id) from error
        return Path(local)

    def cache_dir(self) -> Path:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)

    def file_url(self, repo_id: str, revision: str, path: str) -> str:
        from huggingface_hub import hf_hub_url

        return hf_hub_url(repo_id, path, revision=revision)
