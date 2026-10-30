"""Unified registry for white-box and black-box methods."""

from __future__ import annotations

from secret_keeping.utils.registry import Registry

METHOD_REGISTRY = Registry("METHOD")


def _register_methods() -> None:
    from .blackbox import assistant_prefill, best_of_n, direct_prompt  # noqa: F401
    from .whitebox import jacobian_lens, linear_probe, logit_lens, sae, template_lens  # noqa: F401


def build_method(cfg):
    _register_methods()
    return METHOD_REGISTRY.get(cfg.METHOD.NAME)(cfg)


def method_names() -> list[str]:
    _register_methods()
    return METHOD_REGISTRY.keys()

