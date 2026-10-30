"""Model registry and factory."""

from __future__ import annotations

from secret_keeping.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    # Import registrations only when construction is requested; dry-runs stay lightweight.
    from . import gemma  # noqa: F401

    return MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg).load()

