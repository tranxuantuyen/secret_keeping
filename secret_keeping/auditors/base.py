"""Auditor interface. Auditor protocols are intentionally explicit in outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAuditor(ABC):
    protocol = "unknown"

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def audit(self, results: list) -> list[dict]:
        raise NotImplementedError

