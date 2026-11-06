"""K/V LoRA path switch for DFlash attention (`adapter.kv_lora_path`).

The same `k_proj`/`v_proj` project both the context (target features) and the
noise block. `both` (default, the only vLLM-mergeable variant) adapts both;
`context` / `block` apply the LoRA delta on one path and the frozen base
projection on the other. Installed per instance by switching the class; the
subclass adds one attribute and no parameters.
"""

from speculators.generator.model import AdapterLinear
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention

KV_LORA_PATHS = ("both", "context", "block")


class KVPathDFlashAttention(Qwen3DFlashAttention):
    kv_lora_path: str

    def _project_kv(self, target_hidden, hidden_states):
        context = self.kv_lora_path in {"both", "context"}
        block = self.kv_lora_path in {"both", "block"}

        def project(module, x, adapted):
            if adapted or not isinstance(module, AdapterLinear):
                return module(x)
            return module.base(x)

        return (
            project(self.k_proj, target_hidden, context),
            project(self.k_proj, hidden_states, block),
            project(self.v_proj, target_hidden, context),
            project(self.v_proj, hidden_states, block),
        )


def install_kv_path(drafter, kv_lora_path):
    """Switch every adapted DFlash attention to `kv_lora_path`; return count."""
    if kv_lora_path not in KV_LORA_PATHS:
        raise ValueError(f"Unknown kv_lora_path {kv_lora_path!r}")
    switched = 0
    for layer in drafter.layers:
        attention = layer.self_attn
        adapted = [
            isinstance(getattr(attention, name), AdapterLinear)
            for name in ("k_proj", "v_proj")
        ]
        if not any(adapted):
            continue
        if type(attention) is not Qwen3DFlashAttention:
            raise TypeError(
                f"kv_lora_path needs Qwen3DFlashAttention, got {type(attention)}"
            )
        attention.__class__ = KVPathDFlashAttention
        attention.kv_lora_path = kv_lora_path
        switched += 1
    if not switched:
        raise ValueError(
            f"kv_lora_path={kv_lora_path!r} requires adapted k_proj or v_proj"
        )
    return switched
