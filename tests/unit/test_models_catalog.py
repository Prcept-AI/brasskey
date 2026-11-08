# SPDX-License-Identifier: Apache-2.0
"""The shipped catalog: valid, pinned, permissive, and internally consistent."""

import pytest

from brasskey.engines.types import ModelFormat
from brasskey.models.catalog import UnknownModelError, find_model, load_catalog

CATALOG = load_catalog()
MODELS = CATALOG.models
ARTIFACTS = [(model, artifact) for model in MODELS for artifact in model.artifacts]

# Plausible bits per weight for each quantization (catches a wrong file or parameter count).
BITS_RANGE = {
    "Q4_K_M": (4.5, 5.2),
    "Q8_0": (6.5, 8.6),
    "4bit": (3.2, 4.6),
    "bf16": (15.9, 16.1),
    "fp8": (8.0, 11.5),
    "awq-int4": (4.5, 6.1),
}


def test_catalog_loads_and_ids_are_unique():
    assert len(MODELS) >= 7
    assert len({model.id for model in MODELS}) == len(MODELS)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.id)
def test_every_model_is_permissive_and_runs_everywhere(model):
    assert model.license in {"apache-2.0", "mit"}
    assert model.verified
    formats = {artifact.format for artifact in model.artifacts}
    assert formats == {ModelFormat.GGUF, ModelFormat.MLX, ModelFormat.SAFETENSORS}
    assert any(a.quantization == "bf16" for a in model.artifacts)


@pytest.mark.parametrize(
    ("model", "artifact"), ARTIFACTS, ids=lambda x: getattr(x, "id", None) or x.repo_id
)
def test_artifacts_are_pinned_hashed_and_plausible(model, artifact):
    assert len(artifact.revision) == 40
    assert all(c in "0123456789abcdef" for c in artifact.revision)
    weights = [f for f in artifact.files if f.path.endswith((".gguf", ".safetensors"))]
    assert weights
    assert all(f.sha256 for f in weights), "large files must carry a sha256"
    low, high = BITS_RANGE[artifact.quantization]
    assert low <= model.bits_per_weight(artifact) <= high


def test_quick_start_ranks_are_unique():
    ranks = [model.quick_start_rank for model in MODELS if model.quick_start_rank is not None]
    assert len(ranks) == len(set(ranks))


def test_find_model_is_case_insensitive():
    assert find_model(CATALOG, " Qwen3-4B ").id == "qwen3-4b"


def test_unknown_model_suggests_the_closest_id():
    with pytest.raises(UnknownModelError) as caught:
        find_model(CATALOG, "qwen3-4")
    assert "Did you mean qwen3-4b?" in caught.value.message
    assert "hf:org/name" in (caught.value.fix or "")
