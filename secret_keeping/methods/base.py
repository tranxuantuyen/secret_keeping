"""Small common interface for white-box and black-box elicitation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResponseRecord:
    sample_id: str
    prompt: str
    response: str
    sample_index: int
    target: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResponseRecord":
        return cls(**value)


@dataclass
class MethodResult:
    sample_id: str
    method_name: str
    evidence_type: str
    evidence: dict[str, Any]
    sample_index: int
    target: str | None
    prompt: str
    response: str
    direct_prediction: str | None = None
    scores: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MethodResult":
        return cls(**value)


class BaseMethod(ABC):
    kind = "blackbox"
    evidence_type = "text"
    inference_variant = "standard"
    requires_generation = True

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.METHOD.NAME

    def required_artifacts(self) -> list[str]:
        return []

    def format_input(self, sample, model) -> str:
        return model.format_prompt(sample.prompt)

    def generation_count(self) -> int:
        return self.cfg.INFERENCE.NUM_SAMPLES

    def assemble_responses(self, sample, outputs: list[str]) -> list[ResponseRecord]:
        return [
            ResponseRecord(sample.sample_id, sample.prompt, text, index, sample.target)
            for index, text in enumerate(outputs)
        ]

    @abstractmethod
    def apply(self, record: ResponseRecord, model) -> MethodResult:
        raise NotImplementedError


class WhiteBoxMethod(BaseMethod):
    kind = "whitebox"


class BlackBoxMethod(BaseMethod):
    kind = "blackbox"
