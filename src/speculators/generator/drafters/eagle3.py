"""EAGLE-3 generator family: the generator's original (pre-port) behaviour."""

import torch

from speculators.generator.drafters.base import DrafterFamily
from speculators.losses import resolve_loss_config
from speculators.models.eagle3.data import shift_batch


@DrafterFamily.register("eagle3")
class Eagle3Family(DrafterFamily):
    name = "eagle3"
    score_key = "full_acc_0"
    metric_label = "offline agreement (full_acc_0)"

    def check_drafter(self, drafter) -> None:
        super().check_drafter(drafter)
        if list(drafter.target_layer_ids) != self.bank.hidden_states.layer_ids[:-1]:
            raise ValueError("Drafter and saved target layers differ")

    def call_kwargs(self):
        return {
            "ttt_steps": self.bank.training.ttt_steps,
            "ttt_step_loss_decay": self.bank.training.ttt_step_loss_decay,
            "loss_config": resolve_loss_config(
                self.cfg.optimization.loss_fn or self.bank.training.loss_fn
            ),
        }

    def preprocess(self, sample):
        return shift_batch(sample)

    def trained_length(self, seq_len):
        return seq_len - 1

    def supervised_tokens(self, row):
        return sum(row["loss_mask"][1:])

    def training_noise(self, states):
        noise_std = self.bank.training.noise_std
        if not noise_std:
            return states
        return states + 2 * (torch.rand_like(states) - 0.5) * noise_std

    def identity(self):
        return {
            **{
                key: getattr(self.bank.training, key)
                for key in ("noise_std", "ttt_steps", "ttt_step_loss_decay")
            },
            "loss_fn": self.cfg.optimization.loss_fn or self.bank.training.loss_fn,
        }
