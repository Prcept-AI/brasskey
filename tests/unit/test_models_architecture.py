# SPDX-License-Identifier: Apache-2.0
"""Architecture parsing from config.json and KV-cache math, anchored on known numbers."""

import pytest

from brasskey.models.architecture import (
    ArchitectureError,
    architecture_from_config,
    estimate_active_params,
)
from brasskey.models.types import ModelArchitecture
from fakes.hub import load_config


def parse(name, params=10**9):
    return architecture_from_config(load_config(name), params_total=params)


def test_llama_3_70b_kv_cache_matches_the_inference_notes():
    # 2 x 80 layers x 8 KV heads x 128 x 2 bytes ~ 0.3 MB per token.
    arch = parse("llama-3-70b", 70 * 10**9).architecture
    assert arch.kv_bytes_per_token() == 327_680
    assert arch.kv_bytes(1000) == 327_680_000  # ~0.3 GB for a 1,000-token chat


def test_qwen3_4b():
    parsed = parse("qwen3-4b", 4_022_468_096)
    arch = parsed.architecture
    assert (arch.num_layers, arch.num_kv_heads, arch.head_dim) == (36, 8, 128)
    assert arch.max_context == 40960
    assert arch.kv_bytes_per_token() == 147_456
    assert arch.params_active is None
    assert arch.sliding_layers == 0
    assert parsed.notes == ()


def test_qwen3_moe_active_parameters():
    arch = parse("qwen3-30b-a3b", 30_532_122_624).architecture
    assert arch.is_moe
    assert arch.params_active == pytest.approx(3.35e9, rel=0.01)  # "A3B"


def test_deepseek_v3_mla_cache_and_active_parameters():
    arch = parse("deepseek-v3", 671 * 10**9).architecture
    assert arch.uses_mla
    # One 512-value latent + 64 rotary values per token and layer, 61 layers, 2 bytes each.
    assert arch.kv_bytes_per_token() == (512 + 64) * 2 * 61
    assert 36e9 < (arch.params_active or 0) < 40e9  # published: 37B active


def test_gpt_oss_sliding_window_layers_are_capped():
    arch = parse("gpt-oss-20b", 21 * 10**9).architecture
    assert (arch.sliding_layers, arch.sliding_window) == (12, 128)
    per_layer_token = 2 * 8 * 64 * 2
    assert arch.kv_bytes(8192) == per_layer_token * (12 * 8192 + 12 * 128)
    assert arch.kv_bytes(100) == per_layer_token * 24 * 100  # below the window: all layers grow


def test_qwen_style_use_sliding_window():
    config = {
        **load_config("qwen3-4b"),
        "use_sliding_window": True,
        "sliding_window": 4096,
        "max_window_layers": 28,
    }
    arch = architecture_from_config(config, params_total=10**9).architecture
    assert (arch.sliding_layers, arch.sliding_window) == (8, 4096)


def test_nested_text_config_and_ambiguous_window_errs_large():
    parsed = parse("gemma-3-12b", 12 * 10**9)
    arch = parsed.architecture
    assert (arch.num_layers, arch.num_kv_heads, arch.head_dim) == (48, 8, 256)
    assert arch.sliding_layers == 0  # unknown pattern: assume full attention (upper bound)
    assert any("upper bound" in note for note in parsed.notes)


def test_legacy_field_names():
    arch = parse("gpt2-style").architecture
    assert (arch.num_layers, arch.num_attention_heads, arch.head_dim) == (12, 12, 64)
    assert arch.max_context == 1024


def test_missing_fields_raise_a_clear_error():
    with pytest.raises(ArchitectureError):
        architecture_from_config({"model_type": "mystery"}, params_total=10**9)


def test_missing_context_limit_is_assumed_and_noted():
    config = {
        key: value
        for key, value in load_config("qwen3-4b").items()
        if key != "max_position_embeddings"
    }
    parsed = architecture_from_config(config, params_total=10**9)
    assert parsed.architecture.max_context == 4096
    assert parsed.notes


def test_dense_models_have_no_active_parameter_estimate():
    assert estimate_active_params(load_config("qwen3-4b"), 4 * 10**9) is None


def test_architecture_validation():
    with pytest.raises(ValueError, match="sliding_layers"):
        ModelArchitecture(
            params_total=1,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=1,
            max_context=10,
            sliding_layers=3,
            sliding_window=4,
        )
    with pytest.raises(ValueError, match="sliding_window"):
        ModelArchitecture(
            params_total=1,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=1,
            max_context=10,
            sliding_layers=1,
        )
