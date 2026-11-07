"""Frozen copy of `Qwen3DFlashAttention.forward` before the `_project_kv` seam.

Source: `src/speculators/models/dflash/model_definitions.py` at e7fd5f4
(lines 97-155). Never edit; the S4 test checks that the
seamed forward is bit-identical to this one.
"""
# ruff: noqa: RUF100, TC003  (verbatim frozen copy)

from collections.abc import Callable

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import apply_rotary_pos_emb


def forward(
    self,
    hidden_states: torch.Tensor,
    target_hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Instead of computing the k and v matricies from the hidden states,
    # the target_hidden is injected into the kv cache, (shape is context
    # length + block size)
    bsz, q_len = hidden_states.shape[:-1]
    ctx_len = target_hidden.shape[1]
    q = self.q_proj(hidden_states)
    q = q.view(bsz, q_len, -1, self.head_dim)
    q = self.q_norm(q).transpose(1, 2)
    # This is the main difference from the usual attention mechanism.
    k_ctx = self.k_proj(target_hidden)
    k_noise = self.k_proj(hidden_states)
    v_ctx = self.v_proj(target_hidden)
    v_noise = self.v_proj(hidden_states)
    k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
    # note the length becomes context length + block size
    v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
    k = self.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
    attn_fn: Callable = eager_attention_forward
    if (
        self.config._attn_implementation is not None  # noqa: SLF001
        and self.config._attn_implementation != "eager"  # noqa: SLF001
    ):
        attn_fn = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation  # noqa: SLF001
        ]
    attn_output, attn_weights = attn_fn(
        self,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
