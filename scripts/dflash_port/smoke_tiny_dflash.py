"""S0 smoke: tiny DFlashDraftModel forward + backward on a given device/attention impl.

Usage: python scripts/dflash_port/smoke_tiny_dflash.py --device cuda --attn simple_flex_attention
Fails loudly (non-zero exit) on non-finite loss or missing gradients.
"""

import argparse

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses.utils import resolve_loss_config
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True, choices=["cpu", "cuda"])
    p.add_argument("--attn", required=True, choices=["eager", "sdpa", "simple_flex_attention"])
    p.add_argument("--loss", default="ce")
    p.add_argument("--loss-impl", required=True, choices=["fused", "eager"])
    p.add_argument("--weight", default="dpace", choices=["dpace", "fixed-exp-decay"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    hidden, vocab, n_aux, seq_len = 64, 128, 5, 96
    tl = Qwen3Config(
        hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=vocab,
        _attn_implementation=a.attn,  # type: ignore[call-arg]
    )
    cfg = DFlashSpeculatorConfig(
        transformer_layer_config=tl, draft_vocab_size=vocab, block_size=16,
        aux_hidden_state_layer_ids=list(range(n_aux)), mask_token_id=vocab - 1,
    )
    model = DFlashDraftModel(cfg)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)  # verifier-owned; uninitialised without load_verifier_weights
    torch.nn.init.ones_(model.verifier_norm.weight)
    model = model.to(a.device).train()

    dev = a.device
    out = model(
        hidden_states=torch.randn(1, seq_len, n_aux * hidden, device=dev),
        input_ids=torch.randint(0, vocab - 1, (1, seq_len), device=dev),
        loss_mask=torch.ones(1, seq_len, device=dev),
        verifier_last_hidden_states=torch.randn(1, seq_len, hidden, device=dev),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long, device=dev),
        max_anchors=4,
        loss_config=resolve_loss_config(a.loss, implementation=a.loss_impl),
        per_position_loss_weight=a.weight,
    )
    loss, metrics = out[1], out[2]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {loss}")
    loss.backward()
    trainable = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    missing = [n for n, q in trainable if q.grad is None]
    if missing:
        raise RuntimeError(f"no grad for: {missing}")
    keys = sorted(k for k in metrics if "acc" in k)[:4]
    print(f"OK device={dev} attn={a.attn} loss_fn={a.loss}/{a.loss_impl}/{a.weight} loss={loss.item():.4f} "
          f"n_trainable={len(trainable)} metrics={ {k: float(metrics[k]) for k in keys} }")


if __name__ == "__main__":
    main()
