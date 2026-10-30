"""Dataset registry. Gender and Taboo share the same file-driven loader."""

from __future__ import annotations

from secret_keeping.utils.registry import Registry

from .prompts import PromptDataset

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.register("gender")(PromptDataset)
DATASET_REGISTRY.register("taboo")(PromptDataset)
DATASET_REGISTRY.register("prompts")(PromptDataset)


def build_dataset(cfg) -> PromptDataset:
    return DATASET_REGISTRY.get(cfg.DATA.NAME).from_config(cfg)

