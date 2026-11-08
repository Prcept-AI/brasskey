# SPDX-License-Identifier: Apache-2.0
"""Read a model's architecture from its Hugging Face ``config.json``.

Handles dense and mixture-of-experts models, sliding-window attention (via ``layer_types`` or
Qwen-style ``use_sliding_window``), DeepSeek MLA, and multimodal configs that nest the language
model under ``text_config``. When a config is ambiguous the estimate errs on the large side (full
attention everywhere), and a note says so.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from brasskey.models.types import ModelArchitecture

DEFAULT_MAX_CONTEXT = 4096


class ArchitectureError(ValueError):
    """The config lacks fields needed to estimate memory."""


@dataclass(frozen=True, slots=True)
class ParsedArchitecture:
    architecture: ModelArchitecture
    notes: tuple[str, ...]


def _int(config: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def estimate_active_params(config: Mapping[str, Any], params_total: int) -> int | None:
    """Parameters used per token for mixture-of-experts models; None for dense models.

    Expert weights = MoE layers x experts x 3 matrices x hidden x expert size; a token uses
    ``experts_per_token`` of them. Qwen3-30B-A3B gives ~3.3B, DeepSeek-V3 ~38B (published 37B).
    """
    experts = _int(config, "num_experts", "num_local_experts", "n_routed_experts")
    per_token = _int(config, "num_experts_per_tok")
    hidden = _int(config, "hidden_size")
    expert_size = _int(config, "moe_intermediate_size", "intermediate_size")
    layers = _int(config, "num_hidden_layers")
    if not (experts and per_token and hidden and expert_size and layers) or per_token >= experts:
        return None
    moe_layers = layers - (_int(config, "first_k_dense_replace") or 0)
    expert_params = moe_layers * experts * 3 * hidden * expert_size
    active = params_total - round(expert_params * (1 - per_token / experts))
    return active if 0 < active < params_total else None


def architecture_from_config(config: Mapping[str, Any], *, params_total: int) -> ParsedArchitecture:
    """Build a :class:`ModelArchitecture` from a config.json and the parameter count."""
    nested = config.get("text_config")
    merged: dict[str, Any] = {**config, **nested} if isinstance(nested, Mapping) else dict(config)
    notes: list[str] = []

    layers = _int(merged, "num_hidden_layers", "n_layer", "num_layers")
    heads = _int(merged, "num_attention_heads", "n_head")
    if layers is None or heads is None:
        raise ArchitectureError("config.json has no layer or attention-head count")
    kv_heads = _int(merged, "num_key_value_heads") or heads
    hidden = _int(merged, "hidden_size", "n_embd")
    head_dim = _int(merged, "head_dim") or (hidden // heads if hidden else None)
    if head_dim is None:
        raise ArchitectureError("config.json has no head_dim or hidden_size")
    max_context = _int(merged, "max_position_embeddings", "n_positions", "seq_length")
    if max_context is None:
        max_context = DEFAULT_MAX_CONTEXT
        notes.append(f"Context limit not in config.json; assuming {DEFAULT_MAX_CONTEXT} tokens.")

    window = _int(merged, "sliding_window")
    sliding_layers = 0
    layer_types: object = merged.get("layer_types")
    if isinstance(layer_types, list) and window:
        kinds = cast(list[object], layer_types)
        sliding_layers = sum(1 for kind in kinds if kind == "sliding_attention")
    elif merged.get("use_sliding_window") is True and window:
        full = _int(merged, "max_window_layers") or 0
        sliding_layers = max(layers - full, 0)
    elif window and "use_sliding_window" not in merged:
        notes.append(
            "The config mentions sliding-window attention without saying which layers use it; "
            "the KV cache is estimated as if every layer used the full context (an upper bound)."
        )
    sliding_window = window if sliding_layers else None

    kv_lora_rank = _int(merged, "kv_lora_rank")
    rope_dim = merged.get("qk_rope_head_dim") if kv_lora_rank else None
    architecture = ModelArchitecture(
        params_total=params_total,
        params_active=estimate_active_params(merged, params_total),
        num_layers=layers,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_context=max_context,
        sliding_window=sliding_window,
        sliding_layers=sliding_layers,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim if isinstance(rope_dim, int) else None,
    )
    return ParsedArchitecture(architecture, tuple(notes))
