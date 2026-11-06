"""Drafter families for the LoRA generator (see `base.py`)."""

from speculators.generator.drafters import dflash, eagle3
from speculators.generator.drafters.base import DrafterFamily, drafter_family

__all__ = ["DrafterFamily", "dflash", "drafter_family", "eagle3"]
