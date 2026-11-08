# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helpers: file selection, quantization and format detection, hf: resolution."""

import pytest

from brasskey.engines.types import ModelFormat
from brasskey.models.hub import (
    ModelSourceError,
    RepoFile,
    RepoInfo,
    detect_format,
    detect_quantization,
    parse_hf_spec,
    resolve_hf_model,
    select_files,
)
from fakes.hub import FakeHub, gguf_repo, load_config, safetensors_repo

QUANTS = {"Q8_0": 4_280_000_000, "Q4_K_M": 2_497_280_256, "Q5_K_M": 2_890_000_000}


def test_gguf_prefers_q4_k_m_by_default():
    files, picked = select_files(gguf_repo("o/M-GGUF", None, QUANTS).files, ModelFormat.GGUF)
    assert picked == "Q4_K_M"
    assert [f.path for f in files] == ["M-Q4_K_M.gguf"]


def test_gguf_explicit_quantization_is_case_insensitive():
    files, picked = select_files(
        gguf_repo("o/M-GGUF", None, QUANTS).files, ModelFormat.GGUF, "q8_0"
    )
    assert picked == "Q8_0"
    assert files[0].path == "M-Q8_0.gguf"


def test_gguf_missing_quantization_lists_what_exists():
    with pytest.raises(ModelSourceError) as caught:
        select_files(gguf_repo("o/M-GGUF", None, QUANTS).files, ModelFormat.GGUF, "IQ2_XS")
    assert "M-Q4_K_M.gguf" in caught.value.message


def test_gguf_split_files_are_kept_together():
    files = (
        RepoFile("big-Q4_K_M-00001-of-00002.gguf", 10, "a" * 64),
        RepoFile("big-Q4_K_M-00002-of-00002.gguf", 10, "b" * 64),
        RepoFile("big-Q8_0.gguf", 30, "c" * 64),
    )
    chosen, _ = select_files(files, ModelFormat.GGUF)
    assert len(chosen) == 2


def test_safetensors_keeps_serving_files_only():
    files, _ = select_files(
        safetensors_repo("o/m", 100, params=None).files, ModelFormat.SAFETENSORS
    )
    assert sorted(f.path for f in files) == ["config.json", "model.safetensors", "tokenizer.json"]


def test_safetensors_without_weights_is_an_error():
    with pytest.raises(ModelSourceError):
        select_files((RepoFile("config.json", 1),), ModelFormat.SAFETENSORS)


@pytest.mark.parametrize(
    ("fmt", "config", "expected"),
    [
        (ModelFormat.SAFETENSORS, "qwen3-4b", "bf16"),
        (ModelFormat.SAFETENSORS, "qwen3-4b-awq", "awq-int4"),
        (ModelFormat.SAFETENSORS, "deepseek-v3", "fp8"),
        (ModelFormat.SAFETENSORS, "gpt-oss-20b", "mxfp4"),
        (ModelFormat.MLX, "qwen3-4b-mlx-4bit", "4bit"),
    ],
)
def test_detect_quantization(fmt, config, expected):
    assert detect_quantization(fmt, load_config(config), None) == expected


def test_detect_format():
    assert detect_format(gguf_repo("o/M-GGUF", None, QUANTS)) == ModelFormat.GGUF
    assert detect_format(safetensors_repo("o/m", 1, params=1, tags=("mlx",))) == ModelFormat.MLX
    assert detect_format(safetensors_repo("o/m", 1, params=1)) == ModelFormat.SAFETENSORS
    with pytest.raises(ModelSourceError):
        detect_format(RepoInfo("o/m", "e" * 40, (RepoFile("model.bin", 1),)))


@pytest.mark.parametrize("bad", ["hf:", "hf:noslash", "hf:a/b/c", "hf:/name"])
def test_parse_hf_spec_rejects_bad_names(bad):
    with pytest.raises(ModelSourceError):
        parse_hf_spec(bad)


def test_parse_hf_spec():
    assert parse_hf_spec("hf:Qwen/Qwen3-4B-GGUF:Q8_0") == ("Qwen/Qwen3-4B-GGUF", "Q8_0")
    assert parse_hf_spec("hf:Qwen/Qwen3-4B") == ("Qwen/Qwen3-4B", None)


def gguf_hub(base: str | None = "Qwen/Qwen3-4B"):
    return FakeHub(
        repos={
            "Qwen/Qwen3-4B-GGUF": gguf_repo("Qwen/Qwen3-4B-GGUF", base, QUANTS),
            "Qwen/Qwen3-4B": safetensors_repo("Qwen/Qwen3-4B", 8_044_982_080, params=4_022_468_096),
        },
        configs={"Qwen/Qwen3-4B": load_config("qwen3-4b")},
    )


def test_resolve_gguf_repo_reads_the_base_architecture():
    model = resolve_hf_model("hf:Qwen/Qwen3-4B-GGUF", gguf_hub())
    assert model.id == "hf:Qwen/Qwen3-4B-GGUF"
    assert model.verified is False
    assert model.source_repo == "Qwen/Qwen3-4B"
    assert model.architecture.kv_bytes_per_token() == 147_456
    (artifact,) = model.artifacts
    assert (artifact.format, artifact.quantization) == (ModelFormat.GGUF, "Q4_K_M")


def test_resolve_gguf_without_base_model_explains():
    with pytest.raises(ModelSourceError) as caught:
        resolve_hf_model("hf:Qwen/Qwen3-4B-GGUF", gguf_hub(base=None))
    assert "base model" in caught.value.message


def test_resolve_safetensors_repo_estimates_missing_parameter_count():
    hub = FakeHub(
        repos={"o/m": safetensors_repo("o/m", 8_000_000_000, params=None)},
        configs={"o/m": load_config("qwen3-4b")},
    )
    model = resolve_hf_model("hf:o/m", hub)
    assert model.architecture.params_total == 4_000_000_000
    assert any("estimated from file size" in note for note in model.notes)


def test_resolve_gated_repo_explains():
    gated = RepoInfo(
        "meta/llama", "f" * 40, (RepoFile("model.safetensors", 1, "a" * 64),), gated=True
    )
    with pytest.raises(ModelSourceError) as caught:
        resolve_hf_model("hf:meta/llama", FakeHub(repos={"meta/llama": gated}))
    assert "HF_TOKEN" in (caught.value.fix or "")


def test_gguf_without_a_preferred_quantization_picks_a_single_file():
    files = (
        RepoFile("m-IQ3_XS.gguf", 10, "a" * 64),
        RepoFile("m-IQ2_M.gguf", 9, "b" * 64),
        RepoFile("m-F16-00001-of-00002.gguf", 50, "c" * 64),
    )
    chosen, picked = select_files(files, ModelFormat.GGUF)
    assert picked is None
    assert [file.path for file in chosen] == ["m-IQ2_M.gguf"]  # first by name, not a split part
