# SPDX-License-Identifier: Apache-2.0
"""Run planning: quick start, named models, engines, quantization rules and plain errors."""

import pytest

from brasskey.engines.types import EngineId
from brasskey.models.catalog import UnknownModelError, find_model, load_catalog
from brasskey.models.hub import ModelSourceError
from brasskey.planner import NoFitError, RunRequest, UnsupportedEngineError, plan_run
from brasskey.planner.select import QUICK_START_MAX_DOWNLOAD_BYTES
from fakes import machines
from fakes.hub import FakeHub, gguf_repo, load_config, safetensors_repo

CATALOG = load_catalog()


def plan(hw, **request):
    return plan_run(RunRequest(**request), hw, CATALOG)


# --- quick start (no model named) ---------------------------------------------------------------


def test_quick_start_on_an_18_gb_mac_picks_qwen3_8b_with_mlx():
    result = plan(machines.m3_pro_18gb())
    assert (result.model.id, result.engine, result.fit.artifact.quantization) == (
        "qwen3-8b",
        EngineId.MLX,
        "4bit",
    )
    assert result.fit.comfortable
    assert [option.model_id for option in result.bigger_options] == ["qwen3-14b"]
    assert "most capable quick-start model" in result.reasons[0]
    assert "MLX" in result.reasons[1]


def test_quick_start_on_an_8_gb_mac_picks_a_smaller_model():
    result = plan(machines.m1_8gb())
    assert result.model.id == "qwen3-4b"
    assert result.bigger_options == ()


def test_quick_start_on_cpu_keeps_the_download_small():
    result = plan(machines.cpu_only(32))
    assert (result.model.id, result.engine) == ("qwen3-4b", EngineId.LLAMACPP)
    assert result.fit.artifact.download_bytes <= QUICK_START_MAX_DOWNLOAD_BYTES
    assert "qwen3-8b" in [option.model_id for option in result.bigger_options]


def test_quick_start_counts_already_downloaded_models_as_quick():
    eight_b = next(a for a in find_model(CATALOG, "qwen3-8b").artifacts if a.format == "gguf")
    result = plan_run(
        RunRequest(),
        machines.cpu_only(32),
        CATALOG,
        cached_bytes=lambda artifact: artifact.download_bytes if artifact == eight_b else 0,
    )
    assert result.model.id == "qwen3-8b"


def test_quick_start_on_an_h100_lists_the_big_models():
    result = plan(machines.h100())
    assert result.engine == EngineId.VLLM
    assert result.fit.artifact.download_bytes <= QUICK_START_MAX_DOWNLOAD_BYTES
    assert "qwen3-32b" in [option.model_id for option in result.bigger_options]


def test_quick_start_with_too_long_a_context_offers_a_shorter_one():
    with pytest.raises(NoFitError) as caught:
        plan(machines.cpu_only(3), context_length=32768)  # 2 GiB usable
    error = caught.value
    assert "Qwen3 0.6B doesn't fit here" in error.message
    assert "--context" in (error.fix or "")
    assert error.next_step.command.startswith("brasskey run qwen3-0.6b --dry-run --context")


def test_quick_start_on_a_machine_too_small_for_any_model():
    with pytest.raises(NoFitError) as caught:
        plan(machines.cpu_only(2))  # 1 GiB usable: not even the weights fit
    assert "more memory" in (caught.value.fix or "")


def test_quick_start_when_only_big_downloads_fit():
    cached_nothing = plan_run(RunRequest(), machines.h100(), CATALOG)
    assert cached_nothing.model.id  # sanity: something quick exists on an H100
    with pytest.raises(NoFitError) as caught:
        plan_run(
            RunRequest(),
            machines.h100(),
            CATALOG.model_copy(
                update={
                    "models": tuple(
                        m for m in CATALOG.models if m.id in {"qwen3-32b", "qwen3-30b-a3b"}
                    )
                }
            ),
        )
    assert "more than about 5 GB" in caught.value.message
    assert caught.value.next_step.command.endswith("--dry-run")


# --- named models ---------------------------------------------------------------------------------


def test_named_model_too_big_suggests_the_largest_that_fits():
    with pytest.raises(NoFitError) as caught:
        plan(machines.m3_pro_18gb(), model="qwen3-32b")
    error = caught.value
    assert "doesn't fit here" in error.message
    assert "qwen3-14b" in (error.fix or "")
    assert error.next_step.command == "brasskey run qwen3-14b --dry-run"


def test_named_model_with_too_long_a_context_suggests_a_shorter_one():
    with pytest.raises(NoFitError) as caught:
        plan(machines.m3_pro_18gb(), model="qwen3-14b", context_length=40960)
    fix = caught.value.fix or ""
    assert "--context" in fix
    tokens = int(fix.split("--context ")[1].split()[0])
    assert tokens % 1024 == 0
    assert plan(machines.m3_pro_18gb(), model="qwen3-14b", context_length=tokens).fit.fits


def test_unknown_model_suggests_a_catalog_id():
    with pytest.raises(UnknownModelError):
        plan(machines.m3_pro_18gb(), model="qwen3-4")


def test_more_gpus_get_better_quality():
    quants = [
        plan(machines.l4(count), model="qwen3-32b", engine=EngineId.VLLM).fit.artifact.quantization
        for count in (2, 4, 8)
    ]
    assert quants == ["awq-int4", "fp8", "bf16"]


# --- engines and quantization rules ---------------------------------------------------------------


def test_mlx_is_refused_on_linux():
    with pytest.raises(UnsupportedEngineError) as caught:
        plan(machines.h100(), model="qwen3-4b", engine=EngineId.MLX)
    assert "Apple Silicon" in caught.value.message
    assert "vLLM" in (caught.value.fix or "")


def test_vllm_is_refused_on_a_mac():
    with pytest.raises(UnsupportedEngineError):
        plan(machines.m3_pro_18gb(), model="qwen3-4b", engine=EngineId.VLLM)


def test_old_gpu_falls_back_to_llama_cpp():
    p40 = machines.nvidia("Tesla P40", 24576, (6, 1), 347)
    result = plan(p40, model="qwen3-4b")
    assert result.engine == EngineId.LLAMACPP


def test_t4_skips_fp8_and_prefers_comfortable_awq():
    for engine in (EngineId.VLLM, EngineId.SGLANG):
        result = plan(machines.t4(), model="qwen3-4b", engine=engine)
        assert result.fit.artifact.quantization == "awq-int4"


def test_bf16_on_turing_runs_as_fp16_with_a_note():
    result = plan(machines.t4(), model="qwen3-0.6b", engine=EngineId.VLLM)
    assert result.fit.artifact.quantization == "bf16"
    assert any("float16" in note for note in result.fit.notes)


def test_sglang_needs_ada_or_newer_for_fp8():
    result = plan(machines.l4(), model="qwen3-8b", engine=EngineId.SGLANG)
    assert result.fit.artifact.quantization in {"fp8", "awq-int4"}  # L4 is Ada (8.9): fp8 allowed
    a100 = machines.nvidia("NVIDIA A100-SXM4-40GB", 40960, (8, 0), 1555)
    blocked = plan(a100, model="qwen3-32b", engine=EngineId.SGLANG)
    assert blocked.fit.artifact.quantization != "fp8"


# --- hf: models -----------------------------------------------------------------------------------


def hub():
    return FakeHub(
        repos={
            "Qwen/Qwen3-4B-GGUF": gguf_repo(
                "Qwen/Qwen3-4B-GGUF", "Qwen/Qwen3-4B", {"Q4_K_M": 2_497_280_256}
            ),
            "Qwen/Qwen3-4B": safetensors_repo("Qwen/Qwen3-4B", 8_044_982_080, params=4_022_468_096),
        },
        configs={"Qwen/Qwen3-4B": load_config("qwen3-4b")},
    )


def test_hf_model_is_planned_and_marked_unverified():
    result = plan_run(
        RunRequest(model="hf:Qwen/Qwen3-4B-GGUF"), machines.m3_pro_18gb(), CATALOG, hub=hub()
    )
    assert result.engine == EngineId.LLAMACPP  # GGUF only: MLX has nothing to run
    assert result.model.verified is False
    assert "chosen from Qwen/Qwen3-4B-GGUF" in result.reasons[1]


def test_hf_model_without_files_for_the_requested_engine():
    with pytest.raises(NoFitError) as caught:
        plan_run(
            RunRequest(model="hf:Qwen/Qwen3-4B-GGUF", engine=EngineId.MLX),
            machines.m3_pro_18gb(),
            CATALOG,
            hub=hub(),
        )
    assert "no mlx files" in caught.value.message


def test_hf_model_needs_a_hub():
    with pytest.raises(ModelSourceError):
        plan(machines.m3_pro_18gb(), model="hf:Qwen/Qwen3-4B-GGUF")
