"""Drafter-family interface for the LoRA generator.

A family owns every behaviour that differs between drafter algorithms on the
generator training path (call kwargs, sample preprocessing, supervised-token
counting, training noise, validation score, run identity). Generic code
(`engine`, `data`, `campaign_train`) calls the family instead of hard-coding
one algorithm. Families are keyed by the same `speculators_model_type` strings
as `SpeculatorModel` and registered with `DrafterFamily.register(<type>)`.
"""

from typing import Any, ClassVar

import torch

from speculators.utils.registry import ClassRegistryMixin


class DrafterFamily(ClassRegistryMixin):
    """Per-algorithm hooks; one instance per (experiment config, bank config)."""

    name: ClassVar[str]
    score_key: ClassVar[str]
    metric_label: ClassVar[str]

    def __init__(self, cfg, bank):
        self.cfg, self.bank = cfg, bank

    def check_drafter(self, drafter) -> None:
        """Raise if the loaded frozen drafter does not match this family/bank."""
        model_type = drafter.config.speculators_model_type
        if model_type != self.name:
            raise ValueError(
                f"Drafter checkpoint is {model_type!r}, bank selects family "
                f"{self.name!r}"
            )

    def prepare_drafter(self, drafter) -> None:
        """Family-specific module changes after adapters are installed."""

    def call_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to every drafter forward."""
        raise NotImplementedError

    def preprocess(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-row transform before collation (e.g. EAGLE-3 next-token shift)."""
        raise NotImplementedError

    def trained_length(self, seq_len: int) -> int:
        """Packed length of one row after `preprocess`."""
        raise NotImplementedError

    def supervised_tokens(self, row) -> int:
        """Supervised token count of one bank row, used to weight microbatches."""
        raise NotImplementedError

    def training_noise(self, states: torch.Tensor) -> torch.Tensor:
        """Training-time perturbation of the auxiliary hidden states."""
        raise NotImplementedError

    def identity(self) -> dict[str, Any]:
        """Drafter-training settings recorded in `run_identity`."""
        raise NotImplementedError


def family_name(bank) -> str:  # noqa: ARG001 -- S3 adds the block-drafter branch
    """Family selected by the bank config.

    Bank configs without a block-drafter section train EAGLE-3; the loaded
    checkpoint is checked against this choice in `DrafterFamily.check_drafter`.
    """
    return "eagle3"


def drafter_family(cfg, bank) -> DrafterFamily:
    # Import registers the built-in families.
    import speculators.generator.drafters  # noqa: F401, PLC0415

    name = family_name(bank)
    registry = DrafterFamily.registry or {}
    if name not in registry:
        raise ValueError(
            f"No generator drafter family registered for {name!r}; "
            f"known: {sorted(registry)}"
        )
    return registry[name](cfg, bank)
