"""Auditor registry and factory."""

from __future__ import annotations

from secret_keeping.utils.registry import Registry

AUDITOR_REGISTRY = Registry("AUDITOR")


def build_auditor(cfg):
    from . import direct, llm  # noqa: F401

    return AUDITOR_REGISTRY.get(cfg.AUDITOR.NAME)(cfg)

