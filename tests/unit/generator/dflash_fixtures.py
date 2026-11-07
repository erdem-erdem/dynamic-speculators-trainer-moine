"""Tiny CPU DFlash drafter and block-drafter settings shared by the DFlash tests."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel

AUX = [2, 18, 33]  # BankConfig default hidden_states.layer_ids[:-1]
BLOCK_DRAFTER = {
    "loss_fn": "ce",
    "per_position_loss_weight": "dpace",
    "dflash_decay_gamma": 7.0,
    "dpace_alpha": 0.5,
    "anchor_density": 1 / 16,
    "validation_anchor_seed": 11,
}


def tiny_dflash_drafter(
    *, hidden=32, vocab=64, layers=3, block_size=8, uncompiled=True
):
    """fp32, eager attention; verifier-owned weights initialised (NaN otherwise).

    `uncompiled` mirrors `generator.engine.Runtime.attach_drafter`; pass False
    when the drafter is attached through the Runtime itself.
    """
    tl = Qwen3Config(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        vocab_size=vocab,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl,
        draft_vocab_size=vocab,
        block_size=block_size,
        aux_hidden_state_layer_ids=AUX,
        mask_token_id=vocab - 1,
    )
    model = DFlashDraftModel(config)
    for weight in (model.verifier_lm_head, model.embed_tokens, model.lm_head):
        torch.nn.init.normal_(weight.weight, std=0.5)
    torch.nn.init.ones_(model.verifier_norm.weight)
    if uncompiled:
        forward = type(model).forward
        forward = getattr(forward, "_torchdynamo_orig_callable", forward)
        model.forward = torch.compiler.disable(forward.__get__(model), recursive=False)
    return model.requires_grad_(False).eval()
