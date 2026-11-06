"""DFlash (block diffusion) generator family.

Differences from EAGLE-3 (see `dflash_port/IMPLEMENTATION_PLAN.md` §2, §4):
- adapters on every selected layer (`l{i}_{type}` keys) plus optional `fc`;
- no next-token shift: rows are trained at full length, all loss-mask tokens count;
- anchors (block starts) are sampled per microbatch at a fixed density; validation
  draws them under a fixed seed derived from the microbatch content, so they are
  identical across calls and arms;
- no hidden-state training noise; validation score `full_acc`.
"""

import hashlib
import re
from contextlib import contextmanager, nullcontext

import torch

from speculators.bank.progress import report
from speculators.generator.drafters.base import DrafterFamily
from speculators.losses import resolve_loss_config

ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
_KEY = re.compile(r"l(\d+)_(\w+)")


@DrafterFamily.register("dflash")
class DFlashFamily(DrafterFamily):
    name = "dflash"
    score_key = "full_acc"
    metric_label = "offline agreement (full_acc)"

    def __init__(self, cfg, bank):
        super().__init__(cfg, bank)
        self.settings = bank.training.block_drafter
        if self.settings is None:
            raise ValueError("DFlash family requires bank training.block_drafter")
        if cfg.adapter is None or cfg.target_modules is None:
            raise ValueError("DFlash family requires adapter and target_modules")
        if cfg.optimization.loss_fn not in {None, self.settings.loss_fn}:
            raise ValueError(
                f"optimization.loss_fn={cfg.optimization.loss_fn!r} conflicts with "
                f"block_drafter.loss_fn={self.settings.loss_fn!r}; set it to null"
            )
        if cfg.adapter.kv_lora_path != "both":
            raise NotImplementedError("kv_lora_path other than 'both' lands in S4")
        # Fused losses are Triton kernels; on CPU the numerically matching eager
        # implementation is used (S0 smoke: fused == eager on CUDA).
        self.loss_implementation = (
            "fused" if str(cfg.device).startswith("cuda") else "eager"
        )

    @property
    def head_sharing(self):
        return self.cfg.adapter.head_sharing

    def check_drafter(self, drafter) -> None:
        super().check_drafter(drafter)
        if list(drafter.target_layer_ids) != self.bank.hidden_states.layer_ids[:-1]:
            raise ValueError("Drafter and saved target layers differ")
        if drafter.config.mask_token_id is None:
            raise ValueError("DFlash drafter has no mask_token_id")
        report(
            f"DFlash drafter: layers={len(drafter.layers)}, "
            f"block_size={drafter.block_size}, mask={drafter.config.mask_token_id}, "
            f"sample_from_anchor={drafter.config.sample_from_anchor}, "
            f"loss_implementation={self.loss_implementation}"
        )

    def adapter_targets(self, drafter, target_modules):
        count = len(drafter.layers)
        layers = self.cfg.adapter.layers
        layers = list(range(count)) if layers == "all" else list(layers)
        if any(layer >= count for layer in layers):
            raise ValueError(f"Adapter layers {layers} exceed drafter depth {count}")
        targets = {}
        for layer in layers:
            for name in target_modules:
                if name in ATTENTION_MODULES:
                    block = "self_attn"
                elif name in MLP_MODULES:
                    block = "mlp"
                else:
                    raise ValueError(f"Unsupported DFlash target module {name!r}")
                targets[f"l{layer}_{name}"] = f"layers.{layer}.{block}.{name}"
        if self.cfg.adapter.fc:
            targets["fc"] = "fc"
        return targets

    def generator_layout(self, keys):
        layout = {}
        for key in keys:
            if key == "fc":
                continue
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Unexpected DFlash adapter key {key!r}")
            layout[key] = (match.group(2), int(match.group(1)))
        # `fc` gets its own layer slot after the last adapted layer.
        fc_slot = max((layer for _, layer in layout.values()), default=-1) + 1
        return {key: layout.get(key, ("fc", fc_slot)) for key in keys}

    def call_kwargs(self):
        return {
            "loss_config": resolve_loss_config(
                self.settings.loss_fn, self.loss_implementation
            ),
            "gamma": self.settings.dflash_decay_gamma,
            "per_position_loss_weight": self.settings.per_position_loss_weight,
            "dpace_alpha": self.settings.dpace_alpha,
        }

    def max_anchors(self, packed_length):
        return max(1, round(self.settings.anchor_density * packed_length))

    def microbatch(self, batch, *, training):
        kwargs = {"max_anchors": self.max_anchors(batch["input_ids"].shape[-1])}
        if training:
            return kwargs, nullcontext()
        return kwargs, self._validation_rng(batch)

    @contextmanager
    def _validation_rng(self, batch):
        ids = batch["input_ids"].detach().to("cpu", torch.int64).contiguous()
        content = int.from_bytes(
            hashlib.sha256(ids.numpy().tobytes()).digest()[:8], "little"
        )
        seed = (self.settings.validation_anchor_seed + content) % 2**63
        device = batch["input_ids"].device
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def preprocess(self, sample):
        return sample

    def trained_length(self, seq_len):
        return seq_len

    def supervised_tokens(self, row):
        return sum(row["loss_mask"])

    def training_noise(self, states):
        return states

    def identity(self):
        return {"family": self.name, **self.settings.model_dump(mode="json")}
