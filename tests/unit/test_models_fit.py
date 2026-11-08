# SPDX-License-Identifier: Apache-2.0
"""Memory budgets, fit math, device splits, context limits, warnings and speed ceilings."""

import pytest

from brasskey.engines.types import DevicePool, EngineId, ModelFormat
from brasskey.hardware.types import GIB
from brasskey.models.catalog import find_model, load_catalog
from brasskey.models.fit import estimate_fit
from brasskey.models.types import ArtifactFile, CatalogModel, ModelArchitecture, ModelArtifact
from fakes import machines

CATALOG = load_catalog()


def catalog_fit(model_id, quant, engine, hw, context=8192, concurrency=1, gpu_backends=None):
    model = find_model(CATALOG, model_id)
    artifact = next(a for a in model.artifacts_for(engine) if a.quantization == quant)
    return estimate_fit(
        model,
        artifact,
        engine,
        hw,
        context_length=context,
        concurrency=concurrency,
        gpu_backends=gpu_backends,
    )


def synthetic(weights_bytes, *, fmt=ModelFormat.GGUF, **arch):
    architecture = ModelArchitecture(
        params_total=arch.pop("params_total", 10**9),
        num_layers=arch.pop("num_layers", 1),
        num_attention_heads=arch.pop("num_attention_heads", 8),
        num_kv_heads=arch.pop("num_kv_heads", 1),
        head_dim=arch.pop("head_dim", 1),
        max_context=arch.pop("max_context", 32768),
        **arch,
    )
    suffix = ".gguf" if fmt == ModelFormat.GGUF else ".safetensors"
    artifact = ModelArtifact(
        format=fmt,
        repo_id="test/model",
        revision="c" * 40,
        quantization="Q4_K_M" if fmt == ModelFormat.GGUF else "bf16",
        files=(ArtifactFile(path=f"model{suffix}", size_bytes=weights_bytes, sha256="d" * 64),),
    )
    model = CatalogModel(
        id="test-model",
        display_name="Test",
        family="test",
        license="mit",
        source_repo="test/model",
        architecture=architecture,
        artifacts=(artifact,),
    )
    return model, artifact


def test_qwen3_8b_on_an_18_gb_mac():
    fit = catalog_fit("qwen3-8b", "4bit", EngineId.MLX, machines.m3_pro_18gb())
    assert fit.budget.pool == DevicePool.APPLE_GPU
    assert fit.budget.per_device_bytes == 12 * GIB  # 2/3 of 18 GiB (macOS default, estimate)
    assert fit.budget.is_estimate
    assert fit.kv_cache_bytes == 147_456 * 8192
    expected = fit.weights_bytes + fit.kv_cache_bytes + 3 * GIB // 4 + int(0.05 * fit.weights_bytes)
    assert fit.required_bytes == expected
    assert fit.fits
    assert fit.comfortable
    assert fit.headroom_ratio == pytest.approx(0.49, abs=0.01)
    assert fit.max_context_that_fits == 40960
    # 150 GB/s / ~4.35 GB of weights read per token
    assert fit.est_decode_tokens_per_s == pytest.approx(150e9 / fit.weights_bytes, abs=0.1)


def test_exact_budget_fits_and_one_byte_more_does_not():
    hw = machines.cpu_only(16)
    budget = 16 * GIB - int(16 * GIB * 0.15)  # max(1 GiB, 15%) kept for the system
    kv = 4 * 1024  # 1 layer x 1 KV head x dim 1 x K,V x 2 bytes x 1,024 tokens
    remaining = budget - kv - GIB // 2
    weights = next(
        w
        for w in range(int(remaining / 1.02) - 100, int(remaining / 1.02) + 100)
        if w + int(0.02 * w) == remaining
    )
    model, artifact = synthetic(weights)
    exact = estimate_fit(model, artifact, EngineId.LLAMACPP, hw, context_length=1024)
    assert exact.required_bytes == exact.budget.per_device_bytes == budget
    assert exact.fits
    model, artifact = synthetic(weights + 1)
    assert not estimate_fit(model, artifact, EngineId.LLAMACPP, hw, context_length=1024).fits


def test_max_context_is_the_exact_limit():
    fit = catalog_fit("qwen3-14b", "4bit", EngineId.MLX, machines.m3_pro_18gb(), context=40960)
    assert not fit.fits
    limit = fit.max_context_that_fits
    assert 0 < limit < 40960
    hw = machines.m3_pro_18gb()
    assert catalog_fit("qwen3-14b", "4bit", EngineId.MLX, hw, context=limit).fits
    assert not catalog_fit("qwen3-14b", "4bit", EngineId.MLX, hw, context=limit + 1).fits


def test_concurrency_multiplies_the_kv_cache():
    hw = machines.h100()
    one = catalog_fit("qwen3-8b", "bf16", EngineId.VLLM, hw)
    many = catalog_fit("qwen3-8b", "bf16", EngineId.VLLM, hw, concurrency=8)
    assert many.kv_cache_bytes == 8 * one.kv_cache_bytes


@pytest.mark.parametrize(("gpus", "devices"), [(2, 2), (4, 4), (8, 4)])
def test_tensor_parallel_uses_the_fewest_gpus_that_fit_comfortably(gpus, devices):
    fit = catalog_fit("qwen3-32b", "fp8", EngineId.VLLM, machines.l4(gpus))
    assert fit.devices_used == devices
    if gpus >= 4:
        assert fit.comfortable


def test_tensor_parallel_respects_attention_heads():
    # 6 GPUs: only TP sizes dividing the heads (1, 2, 4) are tried.
    model, artifact = synthetic(
        200 * GIB, fmt=ModelFormat.SAFETENSORS, num_attention_heads=64, num_kv_heads=8
    )
    fit = estimate_fit(model, artifact, EngineId.VLLM, machines.h100(6), context_length=4096)
    assert fit.devices_used == 4


def test_kv_heads_shard_until_tp_exceeds_them_and_mla_replicates():
    hw = machines.h100(8)
    gqa, gqa_art = synthetic(
        500 * GIB,
        fmt=ModelFormat.SAFETENSORS,
        num_attention_heads=64,
        num_kv_heads=4,
        num_layers=80,
        head_dim=128,
    )
    fit = estimate_fit(gqa, gqa_art, EngineId.VLLM, hw, context_length=8192)
    assert fit.devices_used == 8
    kv_per_device = fit.required_per_device_bytes - (500 * GIB // 8) - fit.overhead_bytes // 8
    assert kv_per_device == pytest.approx(fit.kv_cache_bytes / 4, rel=1e-6)

    mla, mla_art = synthetic(
        500 * GIB,
        fmt=ModelFormat.SAFETENSORS,
        num_attention_heads=128,
        num_kv_heads=128,
        num_layers=61,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    fit = estimate_fit(mla, mla_art, EngineId.VLLM, hw, context_length=8192)
    kv_per_device = fit.required_per_device_bytes - (500 * GIB // 8) - fit.overhead_bytes // 8
    assert kv_per_device == pytest.approx(fit.kv_cache_bytes, rel=1e-6)


def test_cpu_budget_respects_container_limits():
    fit = catalog_fit("qwen3-4b", "Q4_K_M", EngineId.LLAMACPP, machines.cpu_only(64, cgroup_gib=8))
    assert fit.budget.pool == DevicePool.CPU
    # Inside a container the OS lives outside the limit: keep only 10% headroom.
    assert fit.budget.per_device_bytes == 8 * GIB - int(8 * GIB * 0.10)


def test_small_machine_keeps_at_least_1_gib_for_the_system():
    fit = catalog_fit("qwen3-0.6b", "Q8_0", EngineId.LLAMACPP, machines.cpu_only(4))
    assert fit.budget.per_device_bytes == 3 * GIB
    assert fit.fits
    assert fit.est_decode_tokens_per_s is None  # no bandwidth figure for system RAM


NO_GPU: frozenset[str] = frozenset()  # an installed build that listed no GPU device


def llama_4b(hw, gpu_backends=None):
    """Qwen3 4B on llama.cpp; ``gpu_backends`` as a build reported them (None: unknown)."""
    return catalog_fit("qwen3-4b", "Q4_K_M", EngineId.LLAMACPP, hw, gpu_backends=gpu_backends)


def test_cpu_only_llama_cpp_on_an_nvidia_machine_is_planned_for_the_cpu():
    hw = machines.h100(ram_gib=64)
    assumed = llama_4b(hw)  # not installed yet: unknown
    cuda = llama_4b(hw, frozenset({"cuda"}))
    cpu = llama_4b(hw, NO_GPU)
    assert cuda == assumed  # a CUDA build is what the plan assumed all along
    assert assumed.budget.pool == DevicePool.NVIDIA_GPU
    assert assumed.warnings == ()
    assert cpu.budget.pool == DevicePool.CPU
    assert "RAM" in cpu.budget.explanation
    assert cpu.est_decode_tokens_per_s is None  # the GPU's bandwidth no longer applies
    (warning,) = cpu.warnings
    assert "reports no GPU it can use" in warning
    assert "Install a CUDA build of llama.cpp or vLLM" in warning  # vLLM runs on Linux


def test_llama_cpp_without_metal_on_a_mac_is_planned_for_the_cpu():
    hw = machines.m3_pro_18gb()
    metal = llama_4b(hw, frozenset({"metal"}))
    cpu = llama_4b(hw, NO_GPU)
    assert metal.budget.pool == DevicePool.APPLE_GPU
    assert cpu.budget.pool == DevicePool.CPU
    assert any("brew reinstall llama.cpp" in warning for warning in cpu.warnings)
    assert not any("Metal" in warning for warning in metal.warnings)


def test_reported_backends_only_change_llama_cpp():
    hw = machines.h100()
    vllm = catalog_fit("qwen3-4b", "bf16", EngineId.VLLM, hw)
    assert catalog_fit("qwen3-4b", "bf16", EngineId.VLLM, hw, gpu_backends=frozenset()) == vllm
    assert llama_4b(machines.cpu_only(32), NO_GPU) == llama_4b(machines.cpu_only(32))  # no warning


def test_low_free_memory_on_a_mac_warns():
    hw = machines.m3_pro_18gb(available_gib=2)
    fit = catalog_fit("qwen3-8b", "4bit", EngineId.MLX, hw)
    assert fit.fits
    assert any("free right now" in warning for warning in fit.warnings)


def test_busy_gpu_warns_that_vllm_cannot_reserve_its_share():
    fit = catalog_fit("qwen3-4b", "bf16", EngineId.VLLM, machines.h100(used_mib=40 * 1024))
    assert any("reserves" in warning for warning in fit.warnings)
    quiet = catalog_fit("qwen3-4b", "bf16", EngineId.VLLM, machines.h100())
    assert quiet.warnings == ()


def test_moe_speed_ceiling_uses_active_parameters():
    fit = catalog_fit("qwen3-30b-a3b", "fp8", EngineId.VLLM, machines.h100())
    model = find_model(CATALOG, "qwen3-30b-a3b")
    share = (model.architecture.params_active or 0) / model.architecture.params_total
    assert fit.est_decode_tokens_per_s == pytest.approx(
        3352e9 / (fit.weights_bytes * share), rel=0.001
    )
    assert any("Mixture of experts" in note for note in fit.notes)


def test_context_above_the_model_limit_is_capped_and_noted():
    fit = catalog_fit("qwen3-4b", "bf16", EngineId.VLLM, machines.h100(), context=131072)
    assert fit.context_length == 40960
    assert any("at most 40,960 tokens" in note for note in fit.notes)
