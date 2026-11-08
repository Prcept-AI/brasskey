# SPDX-License-Identifier: Apache-2.0
"""A fake engine adapter that launches tests/fakes/fake_engine.py, plus a tiny catalog for it."""

import hashlib
import sys
from datetime import date
from pathlib import Path

from brasskey.engines.base import (
    ENGINE_BASE_ENV,
    EngineAdapter,
    EngineInstall,
    InstallHint,
    LaunchSpec,
    LocalModel,
    ServeOptions,
)
from brasskey.engines.llamacpp import LlamaCppAdapter
from brasskey.engines.types import EngineId, ModelFormat
from brasskey.hardware.types import HardwareReport
from brasskey.models.types import (
    ArtifactFile,
    Catalog,
    CatalogModel,
    ModelArchitecture,
    ModelArtifact,
)

FAKE_ENGINE = Path(__file__).resolve().parent / "fake_engine.py"
FAKE_INSTALL = EngineInstall(
    engine=EngineId.LLAMACPP,
    version="0.0.0-fake",
    executable=(sys.executable, str(FAKE_ENGINE)),
    flags=frozenset({"--jinja", "--metrics", "--no-webui", "--offline", "--fit"}),
)


FAKE_MLX_INSTALL = FAKE_INSTALL.model_copy(
    update={"engine": EngineId.MLX, "version": "0.0.0-fake-mlx", "runtime": "uvx"}
)


class FakeAdapter(EngineAdapter):
    """Pretends to be llama.cpp; reuses llama.cpp's failure explanations for realism."""

    id = EngineId.LLAMACPP

    def __init__(
        self,
        mode: str = "ok",
        *,
        load_seconds: float = 0.2,
        startup_timeout_s: float = 10.0,
        child_pid_file: Path | None = None,
        installed: bool = True,
    ) -> None:
        self.mode = mode
        self.load_seconds = load_seconds
        self.startup_timeout_s = startup_timeout_s
        self.child_pid_file = child_pid_file
        self.installed = installed

    def detect(self) -> EngineInstall | None:
        return FAKE_INSTALL if self.installed else None

    def install_hint(self, hw: HardwareReport) -> InstallHint:
        return InstallHint(
            summary="Install the fake engine.", commands=("brew install fake-engine",)
        )

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        argv = [
            *install.executable,
            "--host", options.host,
            "--port", str(options.port),
            "--mode", self.mode,
            "--load-seconds", str(self.load_seconds),
            "--model-name", options.served_model_name,
        ]  # fmt: skip
        if self.child_pid_file is not None:
            argv += ["--child-pid-file", str(self.child_pid_file)]
        base = f"http://{options.host}:{options.port}"
        secret_env = {"LLAMA_API_KEY": options.api_key_env} if options.api_key_env else {}
        return LaunchSpec(
            argv=tuple(argv),
            env=dict(ENGINE_BASE_ENV),
            secret_env=secret_env,
            base_url=f"{base}/v1",
            health_url=f"{base}/health",
            startup_timeout_s=self.startup_timeout_s,
        )

    def explain_failure(self, exit_code, log_tail):  # type: ignore[no-untyped-def]
        return LlamaCppAdapter().explain_failure(exit_code, log_tail)


TINY_CONTENT = {
    "tiny-Q4_K_M.gguf": b"GGUF fake weights for tests " * 64,
    "model.safetensors": b"MLX fake weights for tests " * 64,
}
TINY_MLX_FILES = ("model.safetensors",)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


TINY_ARTIFACT = ModelArtifact(
    format=ModelFormat.GGUF,
    repo_id="test/tiny-GGUF",
    revision="1" * 40,
    quantization="Q4_K_M",
    files=(
        ArtifactFile(
            path="tiny-Q4_K_M.gguf",
            size_bytes=len(TINY_CONTENT["tiny-Q4_K_M.gguf"]),
            sha256=sha256_of(TINY_CONTENT["tiny-Q4_K_M.gguf"]),
        ),
    ),
)

TINY_MLX_ARTIFACT = ModelArtifact(
    format=ModelFormat.MLX,
    repo_id="test/tiny-MLX-4bit",
    revision="2" * 40,
    quantization="4bit",
    files=tuple(
        ArtifactFile(
            path=path, size_bytes=len(TINY_CONTENT[path]), sha256=sha256_of(TINY_CONTENT[path])
        )
        for path in TINY_MLX_FILES
    ),
)

TINY_SAFETENSORS_ARTIFACT = ModelArtifact(
    format=ModelFormat.SAFETENSORS,
    repo_id="test/tiny",
    revision="3" * 40,
    quantization="bf16",
    files=(
        ArtifactFile(
            path="model.safetensors",
            size_bytes=len(TINY_CONTENT["model.safetensors"]),
            sha256=sha256_of(TINY_CONTENT["model.safetensors"]),
        ),
    ),
)

TINY_MODEL = CatalogModel(
    id="tiny",
    display_name="Tiny Test",
    family="test",
    license="apache-2.0",
    source_repo="test/tiny",
    quick_start_rank=1,
    architecture=ModelArchitecture(
        params_total=1_000_000,
        num_layers=2,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        max_context=4096,
    ),
    artifacts=(TINY_ARTIFACT, TINY_MLX_ARTIFACT, TINY_SAFETENSORS_ARTIFACT),
)

TINY_CATALOG = Catalog(generated_at=date(2026, 11, 8), models=(TINY_MODEL,))


class FakeMlxAdapter(FakeAdapter):
    """Pretends to be MLX: no API keys."""

    id = EngineId.MLX
    supports_api_key = False

    def detect(self) -> EngineInstall | None:
        return FAKE_MLX_INSTALL if self.installed else None


FAKE_VLLM_INSTALL = FAKE_INSTALL.model_copy(
    update={
        "engine": EngineId.VLLM,
        "version": "0.32.0-fake",
        "runtime": "docker",
        "image": "vllm/vllm-openai:v0.32.0@sha256:" + "0" * 64,
        "cuda_version": "13.0",
    }
)


class FakeVllmAdapter(FakeAdapter):
    """Pretends to be vLLM in Docker: records the serve options, keeps a container-ish stop."""

    id = EngineId.VLLM

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.options: list[ServeOptions] = []

    def detect(self) -> EngineInstall | None:
        return FAKE_VLLM_INSTALL if self.installed else None

    def build_launch(
        self, install: EngineInstall, model: LocalModel, options: ServeOptions, hw: HardwareReport
    ) -> LaunchSpec:
        self.options.append(options)
        spec = super().build_launch(install, model, options, hw)
        cache = str(model.snapshot_dir.parent.parent)
        return spec.model_copy(
            update={
                "argv": (*spec.argv, "--mount", f"{cache}:/models/tiny:ro"),
                "placeholders": {cache: "{model_cache}"},
            }
        )
