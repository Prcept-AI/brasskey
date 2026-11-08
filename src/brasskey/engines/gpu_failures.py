# SPDX-License-Identifier: Apache-2.0
"""Plain-language explanations for start-up failures of NVIDIA GPU engines (vLLM, SGLang).

Each rule is (log fragments in lower case, must all of them appear?, message, fix), checked in
order: the most specific causes (driver, Docker, permissions) come before generic memory errors.
"""

from collections.abc import Sequence

from brasskey.engines.docker import TOOLKIT_DOCS
from brasskey.errors import ISSUES_URL, BrasskeyError

Rule = tuple[tuple[str, ...], bool, str, str]

SMALLER = "Use a shorter --context or a smaller model (`brasskey run --dry-run` shows what fits)"


def gpu_engine_rules(name: str, *, driver_for_cuda: str, image: str | None) -> tuple[Rule, ...]:
    """Rules shared by vLLM and SGLang (``name`` is the engine's display name)."""
    pull = f"`docker pull {image}`" if image else "the one-time image download"
    return (
        (
            ("unsatisfied condition: cuda", "driver version is insufficient",
             "nvidia driver on your system is too old", "forward compatibility was attempted"),
            False,
            f"The NVIDIA driver is too old for this {name} build.",
            f"Install NVIDIA driver {driver_for_cuda} or newer, then run again (`nvidia-smi` "
            "shows the current one).",
        ),
        (
            ("could not select device driver", "unknown or invalid runtime name: nvidia",
             "nvidia-container-cli: initialization error"),
            False,
            "Docker can't use the NVIDIA GPUs.",
            f"Install the NVIDIA Container Toolkit and restart Docker: {TOOLKIT_DOCS}",
        ),
        (
            ("permission denied while trying to connect to the docker",),
            False,
            "Brasskey can't use Docker: permission denied.",
            "Add your user to the docker group (`sudo usermod -aG docker $USER`), log in again, "
            "then run again.",
        ),
        (
            ("no such image", "unable to find image", "manifest unknown", "pull access denied"),
            False,
            f"The pinned {name} image isn't on this machine.",
            f"Run again and confirm the download, or run {pull}.",
        ),
        (
            ("permission denied", "/models/"),
            True,
            f"The {name} container can't read the model files.",
            "Make the Hugging Face cache readable (`chmod -R a+rX ~/.cache/huggingface/hub`), "
            "then run again.",
        ),
        (
            ("free memory on device", "less than desired gpu memory utilization"),
            False,
            f"Other programs are using GPU memory that {name} needs.",
            "Stop them (see `nvidia-smi`), then run again.",
        ),
        (
            ("maximum number of tokens that can be stored in kv cache",
             "no available memory for the cache blocks",
             "not enough memory. please try to increase",
             "decreasing `max_model_len`", "decrease --context-length"),
            False,
            f"After loading the model, {name} has too little GPU memory left for this context "
            "length.",
            f"{SMALLER}.",
        ),
        (
            ("cuda out of memory", "outofmemoryerror", "out of memory"),
            False,
            f"{name} ran out of GPU memory.",
            f"{SMALLER}, and stop other GPU programs.",
        ),
        (
            ("bfloat16 is only supported on gpus with compute capability",),
            False,
            "This GPU can't run bfloat16 weights, and the engine wasn't told to use float16.",
            f"Please report it at {ISSUES_URL}; meanwhile pick an AWQ model or llama.cpp.",
        ),
        (
            ("nccl error", "ncclsystemerror", "ncclinternalerror", "nccl failure",
             "ncclunhandledcudaerror"),
            False,
            f"{name} couldn't connect its GPUs to each other (NCCL).",
            "Try a model that fits one GPU, or check the GPU links with `nvidia-smi topo -m`.",
        ),
        (
            ("address already in use",),
            False,
            "The engine's port is already in use.",
            "Run again (Brasskey picks another private port).",
        ),
        (
            ("model architectures", "not supported"),
            True,
            f"This {name} version doesn't support this model.",
            "Use another model, or llama.cpp with `--engine llamacpp`.",
        ),
    )  # fmt: skip


def match_failure(rules: Sequence[Rule], log_tail: Sequence[str]) -> BrasskeyError | None:
    text = "\n".join(log_tail).lower()
    for fragments, require_all, message, fix in rules:
        check = all if require_all else any
        if check(fragment in text for fragment in fragments):
            return BrasskeyError(message, fix=fix, code="BK-E-ENGINE-START")
    return None
